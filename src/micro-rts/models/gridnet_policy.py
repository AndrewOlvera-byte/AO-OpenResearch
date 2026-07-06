"""``GridNetPolicy`` — canonical Gym-uRTS GridNet actor-critic (encoder-decoder).

The best-practice policy for full-game MicroRTS (Huang et al. 2021): a conv **encoder**
downsamples the observation to a spatial bottleneck, a conv-transpose **decoder**
upsamples back to the ``H x W`` grid and emits per-cell action logits, and a shared
critic reads the bottleneck. Every unit is commanded every step (``GridNetActionHead``),
with invalid-action masking on each cell — this is what lets masked PPO beat the
scripted competition bots, where the single-unit policy is limited to one action/step.

GroupNorm keeps the forward pass identical during collection and the PPO update.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from core.registry import register

from .shared.GridNetHead import GridNetActionHead
from .shared.MLP import MLP


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class GridNetPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_nvec,
        device: str = "cpu",
        channels: int = 64,
        critic_hidden: tuple[int, ...] = (256,),
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        if not torch.is_tensor(action_nvec):
            action_nvec = torch.as_tensor(list(action_nvec), dtype=torch.long)
        self.grid_cells = int(action_nvec[0])          # H*W (source dim of the full nvec)
        assert self.grid_cells == h * w, "gridnet expects source dim == H*W"
        cell_nvec = action_nvec[1:]                     # per-cell [6,4,4,4,4,7,49]
        self.cell_out = int(cell_nvec.sum())            # 78

        # Encoder: H x W -> H/4 x W/4 spatial bottleneck.
        self.encoder = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), _group_norm(32), nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1, stride=2), _group_norm(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, stride=2), _group_norm(channels), nn.ReLU(),
        )
        # Decoder: bottleneck -> H x W per-cell logits.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
            _group_norm(channels), nn.ReLU(),
            nn.ConvTranspose2d(channels, 32, 4, stride=2, padding=1),
            _group_norm(32), nn.ReLU(),
            nn.Conv2d(32, self.cell_out, 1),
        )
        self.actor = GridNetActionHead(cell_nvec)
        self.critic = MLP(channels * (h // 4) * (w // 4), list(critic_hidden), 1, out_gain=1.0)

        self.device = device
        self.to(device)

    def _bottleneck(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)                         # (N, channels, H/4, W/4)

    def _cell_logits(self, bottleneck: torch.Tensor) -> torch.Tensor:
        d = self.decoder(bottleneck)                     # (N, cell_out, H, W)
        n = d.shape[0]
        return d.permute(0, 2, 3, 1).reshape(n, self.grid_cells, self.cell_out)  # (N, H*W, 78)

    def _value(self, bottleneck: torch.Tensor) -> torch.Tensor:
        return self.critic(bottleneck.flatten(1)).squeeze(-1)

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
             deterministic: bool = False) -> TensorDict:
        z = self._bottleneck(obs.to(self.device))
        mask = mask.to(self.device) if mask is not None else None
        action, logprob = self.actor.sample(self._cell_logits(z), mask, deterministic=deterministic)
        return TensorDict(
            {"action": action, "logprob": logprob, "value": self._value(z)},
            batch_size=[obs.shape[0]],
        )

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor, mask=None):
        """For the PPO update: new logprob, entropy, value of stored actions."""
        z = self._bottleneck(obs)
        logprob, entropy = self.actor.log_prob_entropy(self._cell_logits(z), action, mask)
        return logprob, entropy, self._value(z)

    def freeze_encoder(self, frozen: bool = True) -> None:
        for module in (self.encoder, self.decoder):
            for p in module.parameters():
                p.requires_grad_(not frozen)

    def unfreeze_encoder(self) -> None:
        self.freeze_encoder(False)


@register("model", "cnn_gridnet")
def build_gridnet(obs_shape, action_nvec, device="cpu", **kwargs) -> GridNetPolicy:
    return GridNetPolicy(obs_shape, action_nvec, device=device, **kwargs)
