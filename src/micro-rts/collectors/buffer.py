"""``RolloutBuffer`` — preallocated [T, N] storage on the learner device.

Backed by a single ``TensorDict`` so GAE, device moves, and minibatch views are
one-liners. ``add`` writes one timestep row; ``compute_gae`` fills advantage/return;
``minibatches`` yields flattened [T*N] views for the PPO update.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from tensordict import TensorDict


class RolloutBuffer:
    def __init__(self, horizon, num_envs, obs_shape, action_dim, device="cpu", mask_shape=None):
        self.T, self.N, self.device = horizon, num_envs, device
        # action_dim is an int (single-unit: 8) or a shape tuple (gridnet: (H*W, 7)).
        action_shape = (action_dim,) if isinstance(action_dim, int) else tuple(action_dim)
        fields = {
            "obs": torch.zeros(horizon, num_envs, *obs_shape, device=device),
            "action": torch.zeros(horizon, num_envs, *action_shape, dtype=torch.long, device=device),
            "logprob": torch.zeros(horizon, num_envs, device=device),
            "value": torch.zeros(horizon, num_envs, device=device),
            "reward": torch.zeros(horizon, num_envs, device=device),
            "done": torch.zeros(horizon, num_envs, dtype=torch.bool, device=device),
            "advantage": torch.zeros(horizon, num_envs, device=device),
            "return": torch.zeros(horizon, num_envs, device=device),
        }
        # Invalid-action mask is stored per step so the PPO update masks the exact same
        # logits used at collection time (a valid importance ratio requires it). Stored
        # as bool — for gridnet the (H*W, 79) mask is large, so 1 byte/entry matters.
        self.has_mask = mask_shape is not None
        if self.has_mask:
            fields["mask"] = torch.zeros(horizon, num_envs, *mask_shape, dtype=torch.bool, device=device)
        self.data = TensorDict(fields, batch_size=[horizon, num_envs], device=device)

    def add(self, t, obs, action, logprob, value, reward, done, mask=None) -> None:
        row = dict(obs=obs, action=action, logprob=logprob, value=value, reward=reward, done=done)
        if self.has_mask and mask is not None:
            row["mask"] = mask.bool()
        self.data[t] = TensorDict(row, batch_size=[self.N]).to(self.device)

    @torch.no_grad()
    def compute_gae(self, last_value, gamma=0.99, lam=0.95) -> None:
        adv = torch.zeros(self.N, device=self.device)
        value, reward, done = self.data["value"], self.data["reward"], self.data["done"]
        for t in reversed(range(self.T)):
            nonterminal = (~done[t]).float()
            next_value = last_value if t == self.T - 1 else value[t + 1]
            delta = reward[t] + gamma * next_value * nonterminal - value[t]
            adv = delta + gamma * lam * nonterminal * adv
            self.data["advantage"][t] = adv
        self.data["return"] = self.data["advantage"] + value

    def minibatches(self, num_minibatches, shuffle=True) -> Iterator[TensorDict]:
        flat = self.data.reshape(self.T * self.N)
        order = torch.randperm(self.T * self.N, device=self.device) if shuffle \
            else torch.arange(self.T * self.N, device=self.device)
        for idx in order.chunk(num_minibatches):
            yield flat[idx]
