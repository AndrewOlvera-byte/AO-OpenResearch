"""``CNNPolicy`` — a small conv net satisfying the ``Policy`` protocol.

Deliberately minimal but *real* (genuine GPU forward/backward) so throughput
experiments measure a representative training cost. The MicroRTS action is a flat
MultiDiscrete (one unit-action per env), so the actor head emits ``sum(nvec)`` logits
split per component into independent categoricals.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Categorical


class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, action_nvec, device="cpu"):
        super().__init__()
        c, h, w = obs_shape
        self.splits = action_nvec.tolist()
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        feat = 32 * h * w
        self.actor = nn.Linear(feat, sum(self.splits))
        self.critic = nn.Linear(feat, 1)
        self.device = device
        self.to(device)

    def _logit_chunks(self, obs):
        z = self.encoder(obs)
        return torch.split(self.actor(z), self.splits, dim=-1), z

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> TensorDict:
        chunks, z = self._logit_chunks(obs.to(self.device))
        dists = [Categorical(logits=c) for c in chunks]
        actions = [d.sample() for d in dists]
        logprob = torch.stack([d.log_prob(a) for d, a in zip(dists, actions)], -1).sum(-1)
        return TensorDict(
            {
                "action": torch.stack(actions, -1),
                "logprob": logprob,
                "value": self.critic(z).squeeze(-1),
            },
            batch_size=[obs.shape[0]],
        )

    def evaluate_actions(self, obs, action, mask=None):
        """For the PPO update: new logprob, entropy, value of stored actions.

        ``mask`` is accepted for a uniform Policy API but ignored (unmasked head).
        """
        chunks, z = self._logit_chunks(obs)
        dists = [Categorical(logits=c) for c in chunks]
        logprob = torch.stack([d.log_prob(action[..., i]) for i, d in enumerate(dists)], -1).sum(-1)
        entropy = torch.stack([d.entropy() for d in dists], -1).sum(-1)
        return logprob, entropy, self.critic(z).squeeze(-1)
