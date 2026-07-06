"""``BaseCurriculum`` — the interface a PPO training loop drives.

A curriculum decides, as a function of the global environment-step count, *who the
agent plays against* and *how the model is optimized* (e.g. freezing the encoder
early). The trainer treats it as a black box:

    cur = build("curriculum", **cfg)
    env = MicroRTSVecEnv(cur.env_config(0))
    ...
    events = cur.on_step(global_step, policy)   # each iteration
    if cur.should_rebuild_env(prev_step, global_step):
        env = rebuild(cur.env_config(global_step))

``env_config`` returns an ``EnvConfig`` (environments/microrts_env.py). ``on_step``
returns a small events dict the trainer reacts to (encoder freeze, opponent-pool
refresh, phase name) and logs.
"""

from __future__ import annotations

import abc

from environments.microrts_env import EnvConfig


class BaseCurriculum(abc.ABC):
    @abc.abstractmethod
    def env_config(self, global_step: int) -> EnvConfig: ...

    @abc.abstractmethod
    def phase(self, global_step: int) -> str: ...

    @abc.abstractmethod
    def on_step(self, global_step: int, policy) -> dict: ...

    def should_rebuild_env(self, prev_step: int, global_step: int) -> bool:
        """True when the phase boundary was crossed between the two steps."""
        return self.phase(prev_step) != self.phase(global_step)

    @abc.abstractmethod
    def make_eval_env_config(self) -> EnvConfig: ...
