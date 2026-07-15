"""``AtariActionExpert`` — DreamerV4 actor + critic for a Discrete action space.

Reads a world-model latent state (``n_spatial`` tokens x ``d_latent``), flattens it,
and runs two small MLPs: a **categorical actor** over ``num_actions`` and a **critic**
value with a slow EMA **target critic** for stable bootstrap targets. No action mask
(unlike the MicroRTS GridNet head) — Atari's action set is fully legal every step.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from shared.dreamerv4 import TwoHot

from .config import ActorCriticConfig


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, *, outscale: float | None = None):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU()]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.net = nn.Sequential(*layers)
        if outscale is not None:
            nn.init.xavier_uniform_(self.net[-1].weight)
            self.net[-1].weight.data.mul_(float(outscale))
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class AtariActionExpert(nn.Module):
    def __init__(self, n_spatial: int, d_latent: int, num_actions: int, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.cfg = cfg
        in_dim = n_spatial * d_latent
        self.actor = _MLP(in_dim, list(cfg.hidden), num_actions, outscale=cfg.actor_outscale)
        # Symlog two-hot critic: the MLP emits `critic_bins` logits (zero-init final
        # layer -> initial value 0), decoded to a scalar via the shared coder.
        self.critic = _MLP(in_dim, list(cfg.hidden), cfg.critic_bins)
        nn.init.zeros_(self.critic.net[-1].weight)
        nn.init.zeros_(self.critic.net[-1].bias)
        self.critic_coder = TwoHot(cfg.critic_bins)
        self.target_critic = _MLP(in_dim, list(cfg.hidden), cfg.critic_bins)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    def _logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.actor(z.reshape(z.shape[0], -1))

    def _dist(self, z: torch.Tensor) -> Categorical:
        probs = self._logits(z).softmax(dim=-1)
        if self.cfg.unimix > 0:
            probs = (1.0 - self.cfg.unimix) * probs + self.cfg.unimix / self.num_actions
        return Categorical(probs=probs)

    def value_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z.reshape(z.shape[0], -1))

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic_coder.mean(self.value_logits(z))

    def target_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic_coder.mean(self.target_critic(z.reshape(z.shape[0], -1)))

    def act(self, z: torch.Tensor, mask=None, deterministic: bool = False):
        """Returns (action[N], logprob[N], entropy[N], value[N]). ``mask`` ignored."""
        dist = self._dist(z)
        action = dist.probs.argmax(-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.value(z)

    def evaluate(self, z: torch.Tensor, action: torch.Tensor, mask=None):
        dist = self._dist(z)
        return dist.log_prob(action.long()), dist.entropy(), self.value(z)

    @torch.no_grad()
    def update_target(self, decay: float | None = None) -> None:
        decay = self.cfg.critic_ema if decay is None else decay
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.mul_(decay).add_(p, alpha=1.0 - decay)
