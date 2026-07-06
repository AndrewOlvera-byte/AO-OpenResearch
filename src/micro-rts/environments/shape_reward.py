"""``ShapeReward`` — apply a (possibly time-varying) weight to the raw components.

MicroRTS emits a 6-component reward vector (win, resource, worker, building,
attack, combat-unit). Rather than baking a fixed weight into the JVM env at
construction, this wrapper recomputes the scalar reward in Python from the
surfaced ``raw_rewards`` using a weight vector that the trainer can update every
iteration. That lets the curriculum **anneal dense shaping -> win-focused reward**
over training without rebuilding the (JVM-backed, non-restartable) env.

Stacks *inside* ``NormalizeReward``: base env -> ShapeReward -> NormalizeReward.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .base import VecEnv


class ShapeReward(VecEnv):
    def __init__(self, env: VecEnv, weight) -> None:
        self.env = env
        self.num_envs = env.num_envs
        self.obs_shape = env.obs_shape
        self.action_nvec = env.action_nvec
        self.set_weight(weight)

    def set_weight(self, weight) -> None:
        self.weight = torch.as_tensor(list(weight), dtype=torch.float32)

    def _shape(self, trans: TensorDict) -> TensorDict:
        raw = trans.get("raw_rewards", None)
        if raw is not None:
            trans.set("reward", (raw.float() * self.weight).sum(-1))
        return trans

    def async_reset(self, seed: int | None = None) -> None:
        self.env.async_reset(seed)

    def send(self, actions: torch.Tensor) -> None:
        self.env.send(actions)

    def recv(self) -> TensorDict:
        return self._shape(self.env.recv())

    def close(self) -> None:
        self.env.close()
