"""``DreamCollector`` — fills the Atari ``SequenceReplayBuffer`` from the vec env.

Steps the env with the current policy for ``horizon`` steps, storing
obs/action/reward/cont/is_first. No GAE/value/logprob (those come from imagination),
no mask (Atari has none). Env state persists across calls so episodes continue.
"""

from __future__ import annotations

import torch

from .sequence_buffer import SequenceReplayBuffer


class DreamCollector:
    def __init__(self, env, policy, horizon, buffer: SequenceReplayBuffer | None = None,
                 capacity=None, device="cpu"):
        self.env, self.policy, self.horizon, self.device = env, policy, horizon, device
        self._trans = env.reset()
        if buffer is None:
            buffer = SequenceReplayBuffer(capacity or max(horizon * 4, horizon),
                                          env.num_envs, env.obs_shape, device)
        self.buffer = buffer

    @torch.no_grad()
    def collect(self) -> SequenceReplayBuffer:
        for _ in range(self.horizon):
            trans = self._trans
            obs = trans["obs"].to(self.device)
            out = self.policy.step(obs)
            nxt = self.env.step(out["action"])
            self.buffer.add(
                obs=obs, action=out["action"],
                reward=nxt["reward"].to(self.device),
                cont=(~nxt["done"]).float().to(self.device),
                is_first=trans.get("is_first", torch.zeros(self.env.num_envs, dtype=torch.bool)),
            )
            self._trans = nxt
        return self.buffer
