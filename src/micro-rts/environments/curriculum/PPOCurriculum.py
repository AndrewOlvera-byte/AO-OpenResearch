"""``PPOCurriculum`` — bot-then-self-play schedule for from-scratch PPO.

Phases, by global environment-step count:

- **bot** (``global_step < bot_steps``): play scripted MicroRTS bots (``bots``
  cycled across sub-envs). Optionally **freeze the obs encoder** for the first
  ``freeze_steps`` steps so the policy/value heads can stabilize on top of the
  random-but-fixed features before the encoder starts moving — a common trick for
  learning a deep encoder from scratch under RL.
- **selfplay** (``global_step >= bot_steps``): play snapshots of the agent's own
  past selves drawn from an :class:`OpponentPool`. A fresh snapshot is pushed every
  ``snapshot_every`` steps.

``on_step`` returns the per-iteration events the trainer acts on and logs.
"""

from __future__ import annotations

from environments.microrts_env import EnvConfig
from rewards.rewards import lerp_weights, reward_weight

from core.registry import register

from .BaseCurriculum import BaseCurriculum
from .OpponentPool import OpponentPool


class PPOCurriculum(BaseCurriculum):
    def __init__(
        self,
        bot_steps: int = 2_000_000,
        freeze_steps: int = 200_000,
        bots: tuple[str, ...] = ("randomBiasedAI", "workerRushAI", "lightRushAI", "coacAI"),
        eval_bots: tuple[str, ...] = ("coacAI",),
        map_path: str = "maps/16x16/basesWorkers16x16.xml",
        max_steps: int = 2000,
        num_envs: int = 256,
        reward_preset: str = "dense_shaped",
        reward_start_preset: str = "dense_shaped",
        reward_end_preset: str = "win_focused",
        reward_anneal_steps: int = 4_000_000,
        snapshot_every: int = 200_000,
        pool_capacity: int = 8,
        recency_bias: float = 2.0,
        gridnet: bool = False,
        mix_bot_block: int = 0,
        mix_selfplay_block: int = 0,
    ) -> None:
        self.bot_steps = bot_steps
        self.freeze_steps = freeze_steps
        # Mixed bot+self-play region (after the initial pure-bot warmup of
        # ``bot_steps``): alternate blocks of ``mix_bot_block`` steps vs scripted bots
        # and ``mix_selfplay_block`` steps vs past selves. Keeping the bots (esp. the
        # hardest one) in rotation fights catastrophic forgetting, while self-play
        # pushes past the scripted skill ceiling. Both blocks > 0 enables mixing; if
        # either is 0 the schedule is the classic pure-bot-then-pure-self-play.
        self.mix_bot_block = mix_bot_block
        self.mix_selfplay_block = mix_selfplay_block
        self.bots = tuple(bots)
        self.eval_bots = tuple(eval_bots)
        self.map_path = map_path
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.gridnet = gridnet
        self.reward_preset = reward_preset
        # Reward annealing: bootstrap on dense shaping, then shift toward the true
        # win objective so the policy can't just farm short-horizon shaped reward.
        self.reward_start = reward_weight(reward_start_preset)
        self.reward_end = reward_weight(reward_end_preset)
        self.reward_anneal_steps = reward_anneal_steps
        self.snapshot_every = snapshot_every
        self.pool = OpponentPool(capacity=pool_capacity, recency_bias=recency_bias)
        self._last_snapshot_step = -1

    def reward_weight_vec(self, global_step: int) -> tuple[float, ...]:
        """Current shaping weights, annealed start->end over ``reward_anneal_steps``."""
        t = global_step / self.reward_anneal_steps if self.reward_anneal_steps > 0 else 1.0
        return lerp_weights(self.reward_start, self.reward_end, t)

    # --- schedule --------------------------------------------------------
    def phase(self, global_step: int) -> str:
        if global_step < self.bot_steps:
            return "bot"
        if self.mix_bot_block > 0 and self.mix_selfplay_block > 0:
            cycle = self.mix_bot_block + self.mix_selfplay_block
            pos = (global_step - self.bot_steps) % cycle
            return "bot" if pos < self.mix_bot_block else "selfplay"
        return "selfplay"

    def _encoder_frozen(self, global_step: int) -> bool:
        return global_step < self.freeze_steps

    def env_config(self, global_step: int) -> EnvConfig:
        rw = reward_weight(self.reward_preset)
        if self.phase(global_step) == "selfplay":
            return EnvConfig(
                num_envs=self.num_envs, map_path=self.map_path, max_steps=self.max_steps,
                mode="selfplay", reward_weight=rw, gridnet=self.gridnet,
            )
        return EnvConfig(
            num_envs=self.num_envs, map_path=self.map_path, max_steps=self.max_steps,
            mode="bot", bots=self.bots, reward_weight=rw, gridnet=self.gridnet,
        )

    def make_eval_env_config(self) -> EnvConfig:
        return EnvConfig(
            num_envs=len(self.eval_bots), map_path=self.map_path, max_steps=self.max_steps,
            mode="bot", bots=self.eval_bots, reward_weight=reward_weight(self.reward_preset),
            gridnet=self.gridnet,
        )

    # --- per-iteration hook ----------------------------------------------
    def on_step(self, global_step: int, policy) -> dict:
        events: dict = {
            "phase": self.phase(global_step),
            "encoder_frozen": self._encoder_frozen(global_step),
            "pool_size": len(self.pool),
            "snapshot_pushed": False,
        }
        # Apply encoder freeze/unfreeze (idempotent toggle on the policy).
        if hasattr(policy, "freeze_encoder"):
            policy.freeze_encoder(events["encoder_frozen"])

        # Grow the opponent pool during self-play.
        if events["phase"] == "selfplay":
            due = global_step - self._last_snapshot_step >= self.snapshot_every
            if due or len(self.pool) == 0:
                self.pool.push(policy.state_dict())
                self._last_snapshot_step = global_step
                events["snapshot_pushed"] = True
                events["pool_size"] = len(self.pool)
        return events


@register("curriculum", "ppo")
def build_ppo_curriculum(**kwargs) -> PPOCurriculum:
    return PPOCurriculum(**kwargs)
