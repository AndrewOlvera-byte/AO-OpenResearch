"""``MultiDiscreteActionHead`` — categorical head for MicroRTS actions.

The MicroRTS action is a flat MultiDiscrete (one unit-action per env), so the
head emits ``sum(nvec)`` logits from a feature vector and splits them per
component into independent ``Categorical`` distributions. Keeps all the
action-distribution math out of the policy module.

The engine (gym_microrts 0.3.2) exposes no action mask, so these are plain
categoricals with no invalid-action masking.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from .MLP import orthogonal_init


class MultiDiscreteActionHead(nn.Module):
    def __init__(self, in_dim: int, nvec: torch.Tensor, logits_gain: float = 0.01) -> None:
        super().__init__()
        self.splits: list[int] = nvec.tolist()
        self.logits = nn.Linear(in_dim, int(sum(self.splits)))
        # Small gain -> near-uniform initial policy (PPO best practice).
        orthogonal_init(self.logits, gain=logits_gain)

    def _dists(self, feat: torch.Tensor) -> list[Categorical]:
        chunks = torch.split(self.logits(feat), self.splits, dim=-1)
        return [Categorical(logits=c) for c in chunks]

    def sample(self, feat: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick an action per component. Returns (action[N,A], logprob[N]).

        ``deterministic=True`` takes the argmax (greedy) per component — used for
        evaluation; training always samples.
        """
        dists = self._dists(feat)
        if deterministic:
            actions = [d.logits.argmax(dim=-1) for d in dists]
        else:
            actions = [d.sample() for d in dists]
        logprob = torch.stack([d.log_prob(a) for d, a in zip(dists, actions)], -1).sum(-1)
        return torch.stack(actions, -1), logprob

    def log_prob_entropy(
        self, feat: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate stored actions. Returns (logprob[N], entropy[N])."""
        dists = self._dists(feat)
        logprob = torch.stack(
            [d.log_prob(action[..., i]) for i, d in enumerate(dists)], -1
        ).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], -1).sum(-1)
        return logprob, entropy
