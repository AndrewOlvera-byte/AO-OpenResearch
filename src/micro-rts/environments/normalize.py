"""Reward normalization wrapper (SB3 ``VecNormalize``-style).

MicroRTS uses a 6-component shaped reward whose pieces span a ~50x scale range
(``win=10`` ... ``building=0.2``) and whose terminal win/loss (``+/-10``) is both
huge and rare. Left raw, the value function must fit targets of wildly varying
scale and the rare win spike produces large TD errors / value-loss gradients.

``NormalizeReward`` divides each reward by a running estimate of the standard
deviation of the *discounted return* (not the per-step reward), then clips. This
keeps value targets at ~unit scale and tames the win spike. The policy gradient is
unaffected (advantages are normalized in the PPO loss); this purely stabilizes the
critic and the value/entropy/policy term balance.

This is a thin ``VecEnv`` wrapper, so it is transparent to the Collector/Policy and
only active when the trainer opts in (``training.collector.normalize_reward``).
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .base import VecEnv


class RunningMeanStd:
    """Streaming mean/variance over a stream of scalar samples (Welford / Chan)."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float()
        batch_mean = float(x.mean())
        batch_var = float(x.var(unbiased=False))
        batch_count = x.numel()

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot


class NormalizeReward(VecEnv):
    def __init__(self, env: VecEnv, gamma: float = 0.99, epsilon: float = 1e-8, clip: float = 10.0) -> None:
        self.env = env
        self.gamma, self.epsilon, self.clip = gamma, epsilon, clip
        self.num_envs = env.num_envs
        self.obs_shape = env.obs_shape
        self.action_nvec = env.action_nvec
        self.rms = RunningMeanStd()
        self._returns = torch.zeros(self.num_envs)
        self._is_reset = False

    def async_reset(self, seed: int | None = None) -> None:
        self._returns = torch.zeros(self.num_envs)
        self._is_reset = True
        self.env.async_reset(seed)

    def send(self, actions: torch.Tensor) -> None:
        self._is_reset = False
        self.env.send(actions)

    def recv(self) -> TensorDict:
        trans = self.env.recv()
        if self._is_reset:
            self._is_reset = False
            return trans  # reset frame carries a dummy zero reward; don't normalize
        reward = trans["reward"]
        self._returns = self._returns * self.gamma + reward
        self.rms.update(self._returns)
        std = (torch.tensor(self.rms.var) + self.epsilon).sqrt()
        trans.set("reward", (reward / std).clamp(-self.clip, self.clip))
        # Reset the discounted-return accumulator on episode boundaries.
        self._returns = self._returns * (~trans["done"]).float()
        return trans

    def close(self) -> None:
        self.env.close()
