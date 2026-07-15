"""``MicroRTSVecEnv`` — in-process vectorized MicroRTS env behind the ``VecEnv`` API.

Wraps gym_microrts' ``MicroRTSGridModeVecEnv`` (which already steps N games across
Java threads in one JNI call). Opponent is configured via ``EnvConfig``:

- ``mode="bot"``      : player 2 is a scripted expert; ``bots`` names are cycled
                         across the N envs (e.g. ``("coacAI", "workerRushAI")``).
- ``mode="selfplay"`` : both player slots are agent-controlled. gym_microrts returns
                         consecutive env pairs (the two players), exposed uniformly as
                         ``num_envs`` so the Collector/Policy treat them identically.

Episode auto-reset is handled internally by the Java engine, so this adapter only
encodes obs (NHWC one-hot -> NCHW float) and the action codec (tensor -> int[][][]).

Invalid action masking: gym_microrts 0.3.2 does not expose ``get_action_mask()`` on
the Python wrapper, but the underlying JVM client (``vec_client``) does expose
``getMasks(player)``. It returns an ``(N, H, W, 1 + sum(component_nvec))`` binary
mask per env: channel 0 is the source-unit selectable flag (which cells hold an
owned unit that can act this turn), and channels ``1:`` are the per-component action
masks ``[6,4,4,4,4,7,49]`` for that cell. We surface it (flattened to
``(N, H*W, 79)``) as ``mask`` so the masked policy can restrict the single-unit
action to legal choices — the single most important ingredient for learning on
MicroRTS (Huang & Ontanon, invalid action masking).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDict

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

from .base import VecEnv

# Weights for the 6 raw MicroRTS reward components (win, resource, worker,
# building, attack, combat-unit). See docs/micro-rts/NOTEBOOK.md.
DEFAULT_REWARD_WEIGHT = (10.0, 1.0, 1.0, 0.2, 1.0, 4.0)


@dataclass
class EnvConfig:
    num_envs: int = 8
    map_path: str = "maps/16x16/basesWorkers16x16.xml"
    max_steps: int = 2000
    mode: Literal["bot", "selfplay"] = "bot"
    bots: tuple[str, ...] = ("randomBiasedAI",)
    reward_weight: tuple[float, ...] = DEFAULT_REWARD_WEIGHT
    # GridNet: command *every* cell's unit each step (canonical Gym-uRTS setup) via a
    # per-cell (N, H*W, 7) action, instead of one unit per step. Requires a gridnet
    # policy (model type cnn_gridnet). The env codec prepends the source cell index.
    gridnet: bool = False
    # Surface the scripted bot's gridnet action each step as ``opponent_action``
    # (N, H*W, 7). Requires the patched jar (infra/microrts-jar-patch); off by
    # default because the JNI array conversion costs per-step time that pure RL
    # collection does not need.
    opponent_action: bool = False
    # Surface the Markov-complete structured engine snapshot used by world-model
    # v2: ``full_state`` (N,H*W,16) and ``full_globals`` (N,8), plus terminal
    # versions on done lanes. Requires the patched jar.
    full_state: bool = False
    # Per-lane seat of the *Python-controlled* player in bot mode (0 or 1 each,
    # length num_envs). Seat 1 puts the scripted bot in the player-0 role —
    # role-swapped data collection. None = all 0 (stock behavior). Patched jar only.
    player_ids: tuple[int, ...] | None = None


class MicroRTSVecEnv(VecEnv):
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        rw = np.array(cfg.reward_weight, dtype=np.float64)
        common = dict(max_steps=cfg.max_steps, map_path=cfg.map_path, reward_weight=rw)
        if cfg.mode == "selfplay":
            assert cfg.num_envs % 2 == 0, (
                "selfplay needs an even num_envs (player pairs)"
            )
            self._env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=cfg.num_envs, num_bot_envs=0, ai2s=[], **common
            )
        else:
            ais = [
                getattr(microrts_ai, cfg.bots[i % len(cfg.bots)])
                for i in range(cfg.num_envs)
            ]
            self._env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=0, num_bot_envs=cfg.num_envs, ai2s=ais, **common
            )
        self.num_envs = self._env.num_envs
        self.gridnet = cfg.gridnet
        h, w, c = self._env.observation_space.shape
        self.obs_shape = (c, h, w)
        # Full single-unit nvec [H*W, 6, 4, 4, 4, 4, 7, 49]. Both heads read it:
        # the single-unit head uses it directly; the gridnet head takes nvec[0] as the
        # cell count and nvec[1:] as the per-cell component sizes.
        self.action_nvec = torch.as_tensor(
            self._env.action_space.nvec.tolist(), dtype=torch.long
        )
        self._grid_cells = h * w
        # Mask width per cell: source-selectable flag (1) + per-component action masks.
        self.mask_width = 1 + int(self.action_nvec[1:].sum())
        self.mask_shape = (self._grid_cells, self.mask_width)
        self._pending: TensorDict | None = None
        # Patched microrts.jar (infra/microrts-jar-patch) exposes the pre-reset
        # terminal arrival frame per lane; stock 0.3.2 auto-resets before Python
        # ever sees it. Feature-detected so the wrapper runs on either jar.
        self._has_terminal_obs = hasattr(self._env.vec_client, "terminalObservation")
        self._zero_terminal = (
            torch.zeros((self.num_envs, *self.obs_shape), dtype=torch.float32)
            if self._has_terminal_obs
            else None
        )
        has_opp_patch = hasattr(self._env.vec_client, "opponentAction")
        has_full_state_patch = hasattr(self._env.vec_client, "fullState")
        # Seat-1 owner fix: GameState.getMatrixObservation encodes owner as
        # (unit.player + seat) % 2, so neutral resources (player=-1) come out as
        # "own" when the Python player sits in seat 1. Re-tag resource cells as
        # owner-none so seat-swapped obs match the seat-0 distribution the
        # tokenizer was trained on.
        self._seat1: torch.Tensor | None = None
        if cfg.player_ids is not None and any(p == 1 for p in cfg.player_ids):
            planes = list(self._env.num_planes)  # [hp5, res5, owner3, type, act6]
            owner_off = planes[0] + planes[1]
            self._owner_none_ch = owner_off
            self._owner_own_ch = owner_off + 1
            rid = next(
                t["ID"] for t in self._env.utt["unitTypes"] if t["name"] == "Resource"
            )
            self._resource_ch = owner_off + planes[2] + 1 + rid  # +1: type -1 offset
            self._seat1 = torch.tensor([p == 1 for p in cfg.player_ids])
        if cfg.player_ids is not None:
            assert cfg.mode == "bot", "player_ids only applies to bot mode"
            assert has_opp_patch, "player_ids needs the patched jar (setPlayerIds)"
            assert len(cfg.player_ids) == cfg.num_envs and all(
                p in (0, 1) for p in cfg.player_ids
            ), f"player_ids must be {cfg.num_envs} seats of 0/1"
            import jpype

            self._env.vec_client.setPlayerIds(
                jpype.JArray(jpype.JInt)([int(p) for p in cfg.player_ids])
            )
        self._surface_opp_action = bool(cfg.opponent_action)
        if self._surface_opp_action:
            assert has_opp_patch, (
                "opponent_action=True needs the patched jar (infra/microrts-jar-patch)"
            )
        self._surface_full_state = bool(cfg.full_state)
        if self._surface_full_state:
            assert has_full_state_patch, (
                "full_state=True needs the v2 patched jar (infra/microrts-jar-patch)"
            )

    def _fix_seat1_owner_(self, obs: torch.Tensor) -> torch.Tensor:
        """In-place: re-tag resource cells of seat-1 lanes from owner-own to
        owner-none (see the seat-1 owner note in __init__)."""
        if self._seat1 is not None:
            sub = obs[self._seat1]
            res = sub[:, self._resource_ch].bool()
            sub[:, self._owner_own_ch][res] = 0.0
            sub[:, self._owner_none_ch][res] = 1.0
            obs[self._seat1] = sub
        return obs

    def _encode(self, obs) -> torch.Tensor:
        # (N, H, W, C) one-hot float -> (N, C, H, W) contiguous float32
        out = torch.from_numpy(np.ascontiguousarray(obs)).permute(0, 3, 1, 2).float()
        return self._fix_seat1_owner_(out.contiguous())

    NUM_REWARD_COMPONENTS = 6  # (win, resource, worker, building, attack, combat-unit)

    def _raw_rewards(self, infos) -> torch.Tensor:
        # The engine returns per-env infos each with the *unweighted* 6-component
        # reward vector. Surfacing it lets a shaper reweight in Python (e.g. anneal
        # dense->win-focused) without rebuilding the JVM-backed env.
        if infos is None:
            raw = np.zeros(
                (self.num_envs, self.NUM_REWARD_COMPONENTS), dtype=np.float32
            )
        else:
            raw = np.stack(
                [
                    np.asarray(
                        i.get("raw_rewards", np.zeros(self.NUM_REWARD_COMPONENTS))
                    )
                    for i in infos
                ]
            )
        return torch.as_tensor(raw, dtype=torch.float32)

    def _action_mask(self) -> torch.Tensor:
        # gym_microrts 0.3.2 hides the mask on the Python wrapper but the JVM client
        # exposes getMasks(player). Returns (N, H, W, mask_width) binary; flatten the
        # spatial dims to (N, H*W, mask_width) so cell i aligns with source-unit index
        # i (row-major, same convention as the action's source dimension).
        raw = np.asarray(self._env.vec_client.getMasks(0), dtype=np.float32)
        return torch.from_numpy(raw).reshape(
            self.num_envs, self._grid_cells, self.mask_width
        )

    def _terminal_obs(self, done) -> torch.Tensor:
        # (N, C, H, W) one-hot like ``obs``; zeros except lanes that just
        # terminated, where it holds the true pre-reset terminal frame read from
        # the patched jar's ``terminalObservation``. The zero tensor is shared
        # across steps (never mutated), so the no-done hot path allocates nothing.
        idx = np.flatnonzero(np.asarray(done))
        if idx.size == 0:
            return self._zero_terminal
        out = torch.zeros_like(self._zero_terminal)
        term = self._env.vec_client.terminalObservation
        for i in idx:
            raw = np.asarray(term[int(i)])  # (5, H, W) raw plane encoding
            enc = self._env._encode_obs(raw)  # (H, W, C) one-hot
            out[int(i)] = (
                torch.from_numpy(np.ascontiguousarray(enc)).permute(2, 0, 1).float()
            )
        return self._fix_seat1_owner_(out)

    def _structured_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the current batched v2 engine snapshot from the patched client."""
        state = np.asarray(self._env.vec_client.fullState, dtype=np.int32)
        glob = np.asarray(self._env.vec_client.fullGlobals, dtype=np.int32)
        return torch.from_numpy(state.copy()), torch.from_numpy(glob.copy())

    def _terminal_structured_state(self, done) -> tuple[torch.Tensor, torch.Tensor]:
        """Sparse terminal v2 snapshot; zeros on non-terminal lanes."""
        n = self.num_envs
        state = torch.zeros((n, self._grid_cells, 16), dtype=torch.int32)
        glob = torch.zeros((n, 8), dtype=torch.int32)
        raw_state = self._env.vec_client.terminalFullState
        raw_glob = self._env.vec_client.terminalFullGlobals
        for i in np.flatnonzero(np.asarray(done)):
            state[int(i)] = torch.from_numpy(
                np.asarray(raw_state[int(i)], dtype=np.int32).copy()
            )
            glob[int(i)] = torch.from_numpy(
                np.asarray(raw_glob[int(i)], dtype=np.int32).copy()
            )
        return state, glob

    def _pack(self, obs, reward, done, infos=None) -> TensorDict:
        n = self.num_envs
        td = TensorDict(
            {
                "obs": self._encode(obs),
                "mask": self._action_mask(),
                "reward": torch.as_tensor(np.asarray(reward), dtype=torch.float32),
                "raw_rewards": self._raw_rewards(infos),
                "done": torch.as_tensor(np.asarray(done), dtype=torch.bool),
                "trunc": torch.as_tensor(np.asarray(done), dtype=torch.bool),
                "env_id": torch.arange(n),
            },
            batch_size=[n],
        )
        if self._has_terminal_obs:
            td.set("terminal_obs", self._terminal_obs(done))
        if self._surface_opp_action:
            # Same tick as the submitted action: the bot action that co-produced
            # the returned obs. Zeros right after reset and on self-play lanes.
            raw = np.asarray(self._env.vec_client.opponentAction)
            td.set("opponent_action", torch.from_numpy(raw.astype(np.int64)))
        if self._surface_full_state:
            state, glob = self._structured_state()
            term_state, term_glob = self._terminal_structured_state(done)
            td.set("full_state", state)
            td.set("full_globals", glob)
            td.set("terminal_full_state", term_state)
            td.set("terminal_full_globals", term_glob)
        return td

    def async_reset(self, seed: int | None = None) -> None:
        obs = self._env.reset()
        z = np.zeros(self.num_envs)
        self._pending = self._pack(obs, z, z.astype(bool))

    def _encode_action(self, actions: torch.Tensor) -> np.ndarray:
        a = actions.detach().to("cpu")
        if self.gridnet:
            # Per-cell (N, H*W, 7) -> engine format (N, H*W, 8) by prepending each
            # cell's own index as the source unit. Empty/busy cells carry a NOOP
            # (the masked head forces it) so commanding all cells is a no-op there.
            cells = self._grid_cells
            a = a.view(self.num_envs, cells, -1)
            src = torch.arange(cells).view(1, cells, 1).expand(self.num_envs, -1, 1)
            a = torch.cat([src, a], dim=-1)
        else:
            a = a.view(self.num_envs, -1, len(self.action_nvec))  # (N, num_units=1, 8)
        return a.numpy().astype(np.int32)

    def _encode_gridnet_components(self, actions: torch.Tensor) -> np.ndarray:
        """(N,H*W,7) component actions -> Java GridNet (N,H*W,8)."""
        a = actions.detach().to("cpu").view(self.num_envs, self._grid_cells, -1)
        src = (
            torch.arange(self._grid_cells)
            .view(1, self._grid_cells, 1)
            .expand(self.num_envs, -1, 1)
        )
        return torch.cat([src, a], dim=-1).numpy().astype(np.int32)

    def counterfactual(
        self, actions: torch.Tensor, opponent_actions: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One cloned-engine step from the most recent departure state.

        Call after ``step``: the patched Java client retained that step's exact
        pre-state and the returned opponent action can be held fixed while the
        self action is changed. The live environments are not mutated.
        """
        if not self._surface_full_state:
            raise RuntimeError("counterfactual requires EnvConfig(full_state=True)")
        import jpype

        flags = jpype.JArray(jpype.JBoolean)(valid.detach().cpu().bool().tolist())
        self._env.vec_client.computeCounterfactual(
            self._encode_gridnet_components(actions),
            self._encode_gridnet_components(opponent_actions),
            flags,
        )
        state = np.asarray(self._env.vec_client.counterfactualFullState, dtype=np.int32)
        glob = np.asarray(
            self._env.vec_client.counterfactualFullGlobals, dtype=np.int32
        )
        return torch.from_numpy(state.copy()), torch.from_numpy(glob.copy())

    def send(self, actions: torch.Tensor) -> None:
        obs, reward, done, infos = self._env.step(self._encode_action(actions))
        self._pending = self._pack(obs, reward, done, infos)

    def recv(self) -> TensorDict:
        assert self._pending is not None, "call async_reset()/send() before recv()"
        out, self._pending = self._pending, None
        return out

    def close(self) -> None:
        self._env.close()
