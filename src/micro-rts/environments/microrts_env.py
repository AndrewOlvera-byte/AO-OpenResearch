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


class MicroRTSVecEnv(VecEnv):
    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        rw = np.array(cfg.reward_weight, dtype=np.float64)
        common = dict(max_steps=cfg.max_steps, map_path=cfg.map_path, reward_weight=rw)
        if cfg.mode == "selfplay":
            assert cfg.num_envs % 2 == 0, "selfplay needs an even num_envs (player pairs)"
            self._env = MicroRTSGridModeVecEnv(
                num_selfplay_envs=cfg.num_envs, num_bot_envs=0, ai2s=[], **common
            )
        else:
            ais = [getattr(microrts_ai, cfg.bots[i % len(cfg.bots)]) for i in range(cfg.num_envs)]
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
        self.action_nvec = torch.as_tensor(self._env.action_space.nvec.tolist(), dtype=torch.long)
        self._grid_cells = h * w
        # Mask width per cell: source-selectable flag (1) + per-component action masks.
        self.mask_width = 1 + int(self.action_nvec[1:].sum())
        self.mask_shape = (self._grid_cells, self.mask_width)
        self._pending: TensorDict | None = None

    def _encode(self, obs) -> torch.Tensor:
        # (N, H, W, C) one-hot float -> (N, C, H, W) contiguous float32
        return torch.from_numpy(np.ascontiguousarray(obs)).permute(0, 3, 1, 2).float()

    NUM_REWARD_COMPONENTS = 6  # (win, resource, worker, building, attack, combat-unit)

    def _raw_rewards(self, infos) -> torch.Tensor:
        # The engine returns per-env infos each with the *unweighted* 6-component
        # reward vector. Surfacing it lets a shaper reweight in Python (e.g. anneal
        # dense->win-focused) without rebuilding the JVM-backed env.
        if infos is None:
            raw = np.zeros((self.num_envs, self.NUM_REWARD_COMPONENTS), dtype=np.float32)
        else:
            raw = np.stack([
                np.asarray(i.get("raw_rewards", np.zeros(self.NUM_REWARD_COMPONENTS)))
                for i in infos
            ])
        return torch.as_tensor(raw, dtype=torch.float32)

    def _action_mask(self) -> torch.Tensor:
        # gym_microrts 0.3.2 hides the mask on the Python wrapper but the JVM client
        # exposes getMasks(player). Returns (N, H, W, mask_width) binary; flatten the
        # spatial dims to (N, H*W, mask_width) so cell i aligns with source-unit index
        # i (row-major, same convention as the action's source dimension).
        raw = np.asarray(self._env.vec_client.getMasks(0), dtype=np.float32)
        return torch.from_numpy(raw).reshape(self.num_envs, self._grid_cells, self.mask_width)

    def _pack(self, obs, reward, done, infos=None) -> TensorDict:
        n = self.num_envs
        return TensorDict(
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

    def send(self, actions: torch.Tensor) -> None:
        obs, reward, done, infos = self._env.step(self._encode_action(actions))
        self._pending = self._pack(obs, reward, done, infos)

    def recv(self) -> TensorDict:
        assert self._pending is not None, "call async_reset()/send() before recv()"
        out, self._pending = self._pending, None
        return out

    def close(self) -> None:
        self._env.close()
