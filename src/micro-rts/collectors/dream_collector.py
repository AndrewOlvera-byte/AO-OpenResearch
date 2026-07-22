"""Dreamer collectors — fill a ``SequenceReplayBuffer`` from real MicroRTS play.

``DreamCollector`` steps a single (bot-opponent) ``DreamEnv`` with the current
DreamerV4 policy for ``horizon`` timesteps, storing what the world model needs:
``obs``/``action``/``mask``/``reward``/``cont``/``is_first``. Unlike the PPO
``Collector`` it does **not** compute GAE, value, or log-prob — those are produced
later from imagination (or the online lambda-return losses), not stored rollouts.

``DreamLeagueCollector`` is the Dreamer twin of the PPO league setup
(PPOCurriculum + SelfPlayCollector): it owns a bot env AND a paired self-play env,
alternates blocks of bot play and self-play by collected-step count, and plays the
self-play opponent from an :class:`OpponentPool` of frozen past snapshots
(recency-biased, refreshed every rollout). Both feed the same ``N``-lane buffer —
the self-play env has ``2N`` rows and only the learner rows (``0::2``) are stored.
The first stored row after an env switch is force-tagged ``is_first`` so no
sampled sequence ever blends the two sources, and the pool snapshots exclude the
world model (the opponent only steps ``tokenizer.encode -> action_expert.act``).

The env's running state (``_trans``) is kept across calls so successive collects
continue the same episodes. If the policy's ``step`` returns the latents ``z``
(the DreamerV4 policy does), the collector also maintains a ``WorldModelMemory`` —
the live rolling ``(z, action, is_first)`` context window (reset on env switches).

**Terminal splice** (``terminal_splice=True``): gym_microrts autoresets, so the
frame stored after a ``done`` step is the *next* episode's first frame and the
terminal arrival frame — the only slot whose arrive-aligned continue target is 0
and which carries the win/loss reward — would never enter the buffer. When the
env surfaces the patched jar's ``terminal_obs``, the collector substitutes it in
at the reset slot exactly like the offline ``MRTSSequenceDataset`` splice: obs <-
terminal frame, mask <- 0 (game over, nothing legal), ``is_first`` moves one slot
later. Enable it when the world model is being *trained* on the buffer (hybrid
mode); leave it off when the buffer only feeds the actor with real states (online
mode), since the spliced slot's stored action was chosen at the reset obs, not
the terminal one.
"""

from __future__ import annotations

import torch

from environments.curriculum.OpponentPool import OpponentPool

from .sequence_buffer import SequenceReplayBuffer


class _EnvStream:
    """Persistent stepping state for one env: the running transition plus the
    terminal-splice bookkeeping. ``paired`` envs interleave (learner, opponent)
    rows; ``rows`` slices the learner's share out of a full-env tensor."""

    def __init__(self, env, paired: bool = False):
        self.env, self.paired = env, paired
        self.trans = env.reset()
        self.pending_term = None    # terminal_obs of the step that just ended
        self.pending_done = None    # which learner lanes it applies to
        self.carry_first = None     # lanes whose is_first moved one slot later
        self.force_first = False    # buffer lane switched to this env
        self.ep_return = None       # running per-learner-lane episode return

    def rows(self, x):
        return x[0::2] if self.paired else x

    def on_switch_in(self) -> None:
        """The buffer lane is coming (back) to this env: its next stored row must
        start fresh contexts, and any splice pending from before the switch would
        land against another env's rows — drop it."""
        self.force_first = True
        self.pending_term = self.pending_done = self.carry_first = None


class _DreamCollectorBase:
    """The shared step-and-store core (splice + memory + buffer row)."""

    def __init__(self, policy, horizon, device, memory, terminal_splice):
        self.policy, self.horizon, self.device = policy, horizon, device
        self.memory = memory
        self._splice_wanted = bool(terminal_splice)
        # Real-env episode observability (the failed hybrid run was blind between
        # evals): finished-episode returns + win/loss/timeout counts, drained by
        # the trainer via pop_episode_stats() every console interval.
        self._ep_returns: list[float] = []
        self._wins = self._losses = self._timeouts = 0

    def pop_episode_stats(self) -> dict:
        """Return and reset stats of episodes finished since the last call."""
        if not self._ep_returns:
            return {}
        out = {
            "collect/ep_return": sum(self._ep_returns) / len(self._ep_returns),
            "collect/episodes": float(len(self._ep_returns)),
            "collect/wins": float(self._wins),
            "collect/losses": float(self._losses),
            "collect/timeouts": float(self._timeouts),
        }
        n_dec = self._wins + self._losses + self._timeouts
        if n_dec:
            out["collect/win_rate"] = self._wins / n_dec
        self._ep_returns = []
        self._wins = self._losses = self._timeouts = 0
        return out

    def _track_episodes(self, stream: _EnvStream, nxt) -> None:
        r = stream.rows(nxt["reward"]).detach().float().cpu()
        d = stream.rows(nxt["done"]).detach().bool().cpu()
        if stream.ep_return is None or stream.ep_return.shape != r.shape:
            stream.ep_return = torch.zeros_like(r)
        stream.ep_return += r
        if d.any():
            self._ep_returns.extend(stream.ep_return[d].tolist())
            if "raw_rewards" in nxt.keys():
                win = stream.rows(nxt["raw_rewards"]).detach().cpu()[:, 0][d]
                self._wins += int((win > 0).sum())
                self._losses += int((win < 0).sum())
                self._timeouts += int((win == 0).sum())
            stream.ep_return[d] = 0.0

    def _splice_on(self, stream: _EnvStream) -> bool:
        return self._splice_wanted and "terminal_obs" in stream.trans.keys()

    def _build_buffer(self, stream: _EnvStream, capacity, storage_device):
        probe_mask = stream.trans.get("mask", None)
        n = stream.rows(stream.trans["obs"]).shape[0]
        with torch.no_grad():
            kwargs = {}
            if getattr(self.policy, "uses_structured_state", False):
                kwargs = {
                    "state": stream.rows(stream.trans["full_state"]).to(self.device),
                    "globals_": stream.rows(stream.trans["full_globals"]).to(self.device),
                }
            if getattr(self.policy, "uses_history_state", False):
                kwargs["is_first"] = stream.rows(
                    stream.trans.get(
                        "is_first",
                        torch.ones(stream.env.num_envs, dtype=torch.bool),
                    )
                ).to(self.device)
            out = self.policy.step(
                stream.rows(stream.trans["obs"]).to(self.device),
                stream.rows(probe_mask).to(self.device) if probe_mask is not None else None,
                **kwargs)
        action_shape = tuple(out["action"].shape[1:])
        mask_shape = tuple(stream.rows(probe_mask).shape[1:]) if probe_mask is not None else (0,)
        state_shape = tuple(stream.trans["full_state"].shape[1:]) \
            if "full_state" in stream.trans.keys() else None
        globals_shape = tuple(stream.trans["full_globals"].shape[1:]) \
            if "full_globals" in stream.trans.keys() else None
        return SequenceReplayBuffer(
            capacity or max(self.horizon * 4, self.horizon), n,
            stream.env.obs_shape, action_shape, mask_shape, self.device,
            storage_device=storage_device,
            state_shape=state_shape, globals_shape=globals_shape,
        )

    @torch.no_grad()
    def _collect_step(self, stream: _EnvStream, buffer, opponent=None) -> None:
        trans = stream.trans
        obs_all = trans["obs"].to(self.device)
        mask_all = trans.get("mask", None)
        if mask_all is not None:
            mask_all = mask_all.to(self.device)

        obs = stream.rows(obs_all)
        mask = stream.rows(mask_all) if mask_all is not None else None
        policy_kwargs = {}
        if getattr(self.policy, "uses_structured_state", False):
            policy_kwargs = {
                "state": stream.rows(trans["full_state"]).to(self.device),
                "globals_": stream.rows(trans["full_globals"]).to(self.device),
            }
        if getattr(self.policy, "uses_history_state", False):
            policy_kwargs["is_first"] = stream.rows(
                trans.get(
                    "is_first",
                    torch.zeros(stream.env.num_envs, dtype=torch.bool),
                )
            ).to(self.device)
        out = self.policy.step(obs, mask, **policy_kwargs)
        if stream.paired:
            opp_kwargs = {}
            if getattr(opponent, "uses_structured_state", False):
                opp_kwargs = {
                    "state": trans["full_state"][1::2].to(self.device),
                    "globals_": trans["full_globals"][1::2].to(self.device),
                }
            opp = opponent.step(obs_all[1::2],
                                mask_all[1::2] if mask_all is not None else None,
                                **opp_kwargs)
            actions = torch.empty((stream.env.num_envs, *out["action"].shape[1:]),
                                  dtype=out["action"].dtype, device=out["action"].device)
            actions[0::2] = out["action"]
            actions[1::2] = opp["action"]
        else:
            actions = out["action"]

        n = obs.shape[0]
        raw_first = stream.rows(trans.get("is_first",
                                torch.zeros(stream.env.num_envs, dtype=torch.bool))).bool()

        # Terminal splice: this slot's obs is a fresh reset frame hiding last
        # step's terminal arrival — substitute the true arrival (module docstring).
        obs_store, mask_store, is_first = obs, mask, raw_first
        state_store = stream.rows(trans["full_state"]) if "full_state" in trans.keys() else None
        globals_store = stream.rows(trans["full_globals"]) if "full_globals" in trans.keys() else None
        if stream.carry_first is not None:
            is_first = is_first | stream.carry_first
            stream.carry_first = None
        if stream.force_first:
            is_first = torch.ones_like(is_first)
            stream.force_first = False
        elif self._splice_on(stream) and stream.pending_done is not None:
            sub = raw_first & stream.pending_done
            if sub.any():
                obs_store = obs.clone()
                obs_store[sub] = stream.pending_term[sub].to(obs.dtype).to(self.device)
                if mask is not None:
                    mask_store = mask.clone()
                    mask_store[sub] = 0            # game over: nothing is legal
                is_first = is_first & ~sub
                if state_store is not None:
                    state_store = state_store.clone()
                    globals_store = globals_store.clone()
                    state_store[sub] = stream.rows(trans["terminal_full_state"])[sub]
                    globals_store[sub] = stream.rows(trans["terminal_full_globals"])[sub]
                stream.carry_first = sub           # next slot starts the episode
        stream.pending_done = None

        nxt = stream.env.step(actions)
        self._track_episodes(stream, nxt)
        if self._splice_on(stream) and "terminal_obs" in nxt.keys():
            stream.pending_term = stream.rows(nxt["terminal_obs"])
            stream.pending_done = stream.rows(nxt["done"]).bool()
        # Bot mode gets the engine-executed action from the patched jar.  In
        # paired self-play we already have the exact action submitted for the
        # opponent row.  Keep an explicit validity bit: an all-zero action is a
        # valid NOOP, not an unknown opponent.
        if stream.paired:
            opponent_action = opp["action"]
            opponent_valid = torch.ones(n, dtype=torch.bool, device=self.device)
        elif "opponent_action" in nxt.keys():
            opponent_action = stream.rows(nxt["opponent_action"]).to(self.device)
            opponent_valid = torch.ones(n, dtype=torch.bool, device=self.device)
        else:
            opponent_action = torch.zeros_like(out["action"])
            opponent_valid = torch.zeros(n, dtype=torch.bool, device=self.device)
        buffer.add(
            obs=obs_store,
            action=out["action"],
            opponent_action=opponent_action,
            opponent_valid=opponent_valid,
            mask=mask_store if mask_store is not None else torch.zeros(n, 0),
            reward=stream.rows(nxt["reward"]).to(self.device),
            cont=(~stream.rows(nxt["done"])).float().to(self.device),
            is_first=is_first,
            full_state=state_store,
            full_globals=globals_store,
        )
        if self.memory is not None and "z" in out.keys():
            # Live context keeps the REAL frame semantics (raw is_first).
            self.memory.append(out["z"], out["action"], raw_first)
        stream.trans = nxt


class DreamCollector(_DreamCollectorBase):
    """Single bot-opponent env -> buffer (the non-league path)."""

    def __init__(self, env, policy, horizon, buffer: SequenceReplayBuffer | None = None,
                 capacity=None, device="cpu", memory=None, storage_device="cpu",
                 terminal_splice=False):
        super().__init__(policy, horizon, device, memory, terminal_splice)
        self.env = env
        self._stream = _EnvStream(env, paired=False)
        # Back-compat introspection used by tests/trainer.
        self._trans = self._stream.trans
        self.terminal_splice = self._splice_on(self._stream)
        self.buffer = buffer if buffer is not None else \
            self._build_buffer(self._stream, capacity, storage_device)

    @torch.no_grad()
    def collect(self) -> SequenceReplayBuffer:
        for _ in range(self.horizon):
            self._collect_step(self._stream, self.buffer)
        self._trans = self._stream.trans
        return self.buffer


class DreamLeagueCollector(_DreamCollectorBase):
    """Bot/self-play league collection (see module docstring).

    ``phase()`` mirrors ``PPOCurriculum``: pure bot play for ``bot_steps``
    collected learner transitions, then alternating ``mix_bot_block`` /
    ``mix_selfplay_block``-sized blocks (a zero selfplay block means bots
    forever). Snapshots of the learner (minus the world model) are pushed every
    ``snapshot_every`` steps; each self-play rollout re-samples the opponent.
    """

    def __init__(self, bot_env, sp_env, policy, opponent, horizon, *,
                 capacity=None, device="cpu", memory=None, storage_device="cpu",
                 terminal_splice=False, bot_steps=0, mix_bot_block=0,
                 mix_selfplay_block=0, snapshot_every=250_000, pool_capacity=8,
                 recency_bias=2.0):
        assert sp_env.num_envs == 2 * bot_env.num_envs, \
            "self-play env needs 2x the bot env's lanes (learner/opponent pairs)"
        super().__init__(policy, horizon, device, memory, terminal_splice)
        self.opponent = opponent
        self.bot = _EnvStream(bot_env, paired=False)
        self.sp = _EnvStream(sp_env, paired=True)
        self.bot_steps = int(bot_steps)
        self.mix_bot_block = int(mix_bot_block)
        self.mix_selfplay_block = int(mix_selfplay_block)
        self.snapshot_every = int(snapshot_every)
        self.pool = OpponentPool(capacity=pool_capacity, recency_bias=recency_bias)
        self._steps = 0                     # learner transitions collected so far
        self._last_snapshot = 0
        self._last_phase = None
        self.buffer = self._build_buffer(self.bot, capacity, storage_device)

    def phase(self, steps: int | None = None) -> str:
        s = self._steps if steps is None else steps
        if s < self.bot_steps or self.mix_selfplay_block <= 0:
            return "bot"
        pos = (s - self.bot_steps) % max(self.mix_bot_block + self.mix_selfplay_block, 1)
        return "bot" if pos < self.mix_bot_block else "selfplay"

    @property
    def pool_size(self) -> int:
        return len(self.pool)

    def _push_snapshot(self) -> None:
        sd = {k: v for k, v in self.policy.state_dict().items()
              if not k.startswith("world_model.")}
        self.pool.push(sd)
        self._last_snapshot = self._steps

    @torch.no_grad()
    def collect(self) -> SequenceReplayBuffer:
        ph = self.phase()
        stream = self.sp if ph == "selfplay" else self.bot
        if ph != self._last_phase:
            if self._last_phase is not None:
                stream.on_switch_in()
                if self.memory is not None:
                    self.memory.reset()
            self._last_phase = ph
        if ph == "selfplay":
            if len(self.pool) == 0:
                self._push_snapshot()       # seed: first opponent is current self
            snap = self.pool.sample()
            self.opponent.load_state_dict(snap, strict=False)
        for _ in range(self.horizon):
            self._collect_step(stream, self.buffer,
                               opponent=self.opponent if stream.paired else None)
        self._steps += self.horizon * self.buffer.N
        if self._steps - self._last_snapshot >= self.snapshot_every:
            self._push_snapshot()
        return self.buffer
