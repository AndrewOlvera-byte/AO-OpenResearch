"""``SequenceReplayBuffer`` — ring buffer of ``(T_cap, N)`` Atari transitions.

Same ring-buffer / contiguous ``(B, T)`` sampling as the MicroRTS version, but with
a scalar ``Discrete`` action and no action mask: fields are obs / action / reward /
cont / is_first. Sampling always draws from the most recent ``T_cap`` steps.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class SequenceReplayBuffer:
    def __init__(self, capacity, num_envs, obs_shape, device="cpu"):
        self.T_cap, self.N, self.device = int(capacity), int(num_envs), device
        self.count = 0
        self.data = TensorDict(
            {
                "obs": torch.zeros(self.T_cap, self.N, *obs_shape, device=device),
                "action": torch.zeros(self.T_cap, self.N, dtype=torch.long, device=device),
                "reward": torch.zeros(self.T_cap, self.N, device=device),
                "cont": torch.zeros(self.T_cap, self.N, device=device),
                "is_first": torch.zeros(self.T_cap, self.N, dtype=torch.bool, device=device),
            },
            batch_size=[self.T_cap, self.N],
            device=device,
        )

    @property
    def size(self) -> int:
        return min(self.count, self.T_cap)

    @property
    def num_transitions(self) -> int:
        return self.size * self.N

    def add(self, *, obs, action, reward, cont, is_first) -> None:
        phys = self.count % self.T_cap
        row = TensorDict(
            {"obs": obs, "action": action.long().view(self.N), "reward": reward,
             "cont": cont, "is_first": is_first.bool()},
            batch_size=[self.N],
        ).to(self.device)
        self.data[phys] = row
        self.count += 1

    def can_sample(self, seq_len: int) -> bool:
        return self.size >= seq_len

    def _sample_once(self, batch: int, seq_len: int) -> TensorDict:
        assert self.can_sample(seq_len), "not enough transitions to sample a sequence"
        lo = max(0, self.count - self.T_cap)
        hi = self.count - seq_len
        starts = torch.randint(lo, hi + 1, (batch,), device=self.device)
        envs = torch.randint(0, self.N, (batch,), device=self.device)
        offs = torch.arange(seq_len, device=self.device)
        phys = (starts[:, None] + offs[None, :]) % self.T_cap
        return self.data[phys, envs[:, None]]

    def sample(self, batch: int, seq_len: int, *, avoid_cross_episode: bool = True) -> TensorDict:
        if not avoid_cross_episode:
            return self._sample_once(batch, seq_len)

        chunks = []
        remaining = int(batch)
        for _ in range(16):
            cand = self._sample_once(max(remaining * 2, remaining), seq_len)
            valid = ~cand["is_first"][:, 1:].any(dim=1)
            if bool(valid.any()):
                picked = cand[valid][:remaining]
                chunks.append(picked)
                remaining -= picked.batch_size[0]
                if remaining <= 0:
                    break
        if chunks and remaining <= 0:
            return torch.cat(chunks, dim=0)
        return self._sample_once(batch, seq_len)
