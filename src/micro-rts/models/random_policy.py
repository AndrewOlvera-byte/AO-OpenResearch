"""``RandomPolicy`` — a pass-through test model.

Samples uniform random actions per MultiDiscrete component (the engine ignores
invalid actions), reports the matching constant log-prob, and a zero value. Useful
as a throughput baseline and to exercise the collection stack before a real net.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class RandomPolicy:
    def __init__(self, action_nvec: torch.Tensor, device: str | torch.device = "cpu"):
        self.device = device
        self.nvec = action_nvec.to(device)
        self._logp = -torch.log(self.nvec.float()).sum()  # uniform over all components

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> TensorDict:
        n = obs.shape[0]
        action = (torch.rand(n, len(self.nvec), device=self.device) * self.nvec).long()
        return TensorDict(
            {
                "action": action,
                "logprob": self._logp.expand(n).clone(),
                "value": torch.zeros(n, device=self.device),
            },
            batch_size=[n],
        )
