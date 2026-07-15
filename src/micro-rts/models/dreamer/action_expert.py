"""``ActionExpert`` — Dreamer 4 "action expert": actor + critic over latent states.

Reuses the **GridNet decoder + action head** (the canonical Gym-uRTS policy) but
reads a *world-model latent state* instead of a raw observation: the ``n_spatial``
latent tokens are reshaped to the ``d_latent x H/4 x W/4`` bottleneck and
conv-transposed back to the ``H x W`` grid, exactly like ``GridNetPolicy`` — so the
policy the world model improves is architecturally the same one that plays real
MicroRTS.

Two heads sit on the decoded per-cell features:

- **action logits** (``sum(cell_nvec) = 78`` per cell) -> ``GridNetActionHead`` with
  invalid-action masking.

The action **mask** comes from outside the actor: the engine mask in the real env,
or the tokenizer's *predicted* mask (``GridTokenizer.decode_mask``) inside
imagination. If no mask is supplied at all the head falls back to all-valid.

The **critic** reads the flat latent and regresses the lambda-return; a slow EMA
copy (``target_critic``) provides bootstrap targets (Dreamer's stable value target).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.shared.GridNetHead import GridNetActionHead
from models.shared.MLP import MLP

from shared.dreamerv4 import TwoHot

from .config import ActorCriticConfig


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class ActionExpert(nn.Module):
    def __init__(self, obs_shape, action_nvec, n_spatial: int, d_latent: int,
                 cfg: ActorCriticConfig) -> None:
        super().__init__()
        _, h, w = obs_shape
        if not torch.is_tensor(action_nvec):
            action_nvec = torch.as_tensor(list(action_nvec), dtype=torch.long)
        self.grid_cells = int(action_nvec[0])
        assert self.grid_cells == h * w, "action expert expects source dim == H*W"
        cell_nvec = action_nvec[1:]
        self.cell_out = int(cell_nvec.sum())               # 78
        self.mask_width = 1 + self.cell_out                # 79
        self.n_spatial = n_spatial
        self.d_latent = d_latent
        self.bottleneck_hw = (h // 4, w // 4)
        self.cfg = cfg

        ch = cfg.dec_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(d_latent, ch, 1), _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
        )
        self.action_conv = nn.Conv2d(ch, self.cell_out, 1)
        self.actor = GridNetActionHead(cell_nvec)

        # Symlog two-hot critic: the MLP emits `critic_bins` logits (out_gain=0 ->
        # initial value 0), decoded to a scalar via the shared coder.
        self.critic = MLP(n_spatial * d_latent, list(cfg.critic_hidden), cfg.critic_bins, out_gain=0.0)
        self.critic_coder = TwoHot(cfg.critic_bins)
        self.target_critic = MLP(n_spatial * d_latent, list(cfg.critic_hidden), cfg.critic_bins, out_gain=0.0)
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

    # --- latent -> per-cell logits ---------------------------------------
    def _cell_logits(self, z: torch.Tensor) -> torch.Tensor:
        """z (N,n_spatial,d_latent) -> per-cell action logits (N,H*W,cell_out)."""
        n = z.shape[0]
        hb, wb = self.bottleneck_hw
        m = z.reshape(n, hb, wb, self.d_latent).permute(0, 3, 1, 2)
        feat = self.decoder(m)
        return self.action_conv(feat).permute(0, 2, 3, 1).reshape(n, self.grid_cells, self.cell_out)

    def _ones_mask(self, z: torch.Tensor) -> torch.Tensor:
        return z.new_ones(z.shape[0], self.grid_cells, self.mask_width)

    def value_logits(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic(z.reshape(z.shape[0], -1))

    def value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic_coder.mean(self.value_logits(z))

    def target_value(self, z: torch.Tensor) -> torch.Tensor:
        return self.critic_coder.mean(self.target_critic(z.reshape(z.shape[0], -1)))

    def act(self, z: torch.Tensor, mask: torch.Tensor | None = None,
            deterministic: bool = False):
        """Sample an action for latent ``z`` (N,n_spatial,d_latent).

        Returns (action, logprob, entropy, value). ``mask`` is the engine mask (real
        env) or the tokenizer-predicted mask (imagination); all-valid if None. The
        returned logprob/entropy/value keep grads for the policy/critic losses."""
        cell_logits = self._cell_logits(z)
        if mask is None:
            mask = self._ones_mask(z)
        with torch.no_grad():
            action, _ = self.actor.sample(cell_logits, mask, deterministic=deterministic)
        logprob, entropy = self.actor.log_prob_entropy(cell_logits, action, mask)
        return action, logprob, entropy, self.value(z)

    def evaluate(self, z: torch.Tensor, action: torch.Tensor, mask: torch.Tensor | None = None):
        cell_logits = self._cell_logits(z)
        if mask is None:
            mask = self._ones_mask(z)
        logprob, entropy = self.actor.log_prob_entropy(cell_logits, action, mask)
        return logprob, entropy, self.value(z)

    @torch.no_grad()
    def update_target(self, decay: float | None = None) -> None:
        decay = self.cfg.critic_ema if decay is None else decay
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.mul_(decay).add_(p, alpha=1.0 - decay)
