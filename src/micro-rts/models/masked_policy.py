"""``MaskedActorPolicy`` — spatial conv trunk -> invalid-action-masked actor + critic.

The masked counterpart to ``CNNMLPPolicy``. Two deliberate differences make it the
right policy for MicroRTS's single-unit masked action space:

- **Resolution-preserving trunk.** Selecting *which cell's unit* to command is an
  inherently spatial decision, so the trunk keeps the full ``H*W`` grid (stride-1
  conv residual blocks, GroupNorm) instead of global-average-pooling to a vector
  like the ResNet encoder. ``MaskedMultiDiscreteActionHead`` then reads a per-cell
  source logit from that map.
- **Engine action masking.** ``step``/``evaluate_actions`` thread the
  ``(N, H*W, 79)`` mask surfaced by ``MicroRTSVecEnv`` into the head, restricting
  the policy to legal source cells and legal per-unit action components.

GroupNorm (not BatchNorm) keeps the forward pass identical during rollout
collection and the PPO update — the same train/collect-consistency reason the
ResNet encoder uses it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from core.registry import register

from .shared.MaskedActionHead import MaskedMultiDiscreteActionHead
from .shared.MLP import MLP


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class _ResConvBlock(nn.Module):
    """Stride-1 pre-activation-ish residual block; preserves H x W."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = _group_norm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.relu(x + y)


class MaskedActorPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_nvec,
        device: str = "cpu",
        trunk_channels: int = 64,
        trunk_blocks: int = 4,
        critic_hidden: tuple[int, ...] = (256,),
    ) -> None:
        super().__init__()
        c, h, w = obs_shape
        if not torch.is_tensor(action_nvec):
            action_nvec = torch.as_tensor(list(action_nvec), dtype=torch.long)
        assert int(action_nvec[0]) == h * w, (
            f"source-unit dim {int(action_nvec[0])} must equal H*W={h * w}; "
            "MaskedActorPolicy expects the single-unit MicroRTS action space"
        )

        self.stem = nn.Sequential(
            nn.Conv2d(c, trunk_channels, 3, padding=1), _group_norm(trunk_channels), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[_ResConvBlock(trunk_channels) for _ in range(trunk_blocks)])
        self.actor = MaskedMultiDiscreteActionHead(trunk_channels, action_nvec)
        self.critic = MLP(trunk_channels, list(critic_hidden), 1, out_gain=1.0)

        self.device = device
        self.to(device)

    def _feat_map(self, obs: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(obs))

    def _value(self, feat_map: torch.Tensor) -> torch.Tensor:
        return self.critic(feat_map.mean(dim=(-2, -1))).squeeze(-1)  # global-avg-pool critic

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
             deterministic: bool = False) -> TensorDict:
        feat_map = self._feat_map(obs.to(self.device))
        mask = mask.to(self.device) if mask is not None else None
        action, logprob = self.actor.sample(feat_map, mask, deterministic=deterministic)
        return TensorDict(
            {"action": action, "logprob": logprob, "value": self._value(feat_map)},
            batch_size=[obs.shape[0]],
        )

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor, mask=None):
        """For the PPO update: new logprob, entropy, value of stored actions."""
        feat_map = self._feat_map(obs)
        logprob, entropy = self.actor.log_prob_entropy(feat_map, action, mask)
        return logprob, entropy, self._value(feat_map)

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Toggle trunk grads — the curriculum freezes it early to stabilize."""
        for p in self.stem.parameters():
            p.requires_grad_(not frozen)
        for p in self.blocks.parameters():
            p.requires_grad_(not frozen)

    def unfreeze_encoder(self) -> None:
        self.freeze_encoder(False)


@register("model", "cnn_mlp_masked")
def build_masked_actor(obs_shape, action_nvec, device="cpu", **kwargs) -> MaskedActorPolicy:
    return MaskedActorPolicy(obs_shape, action_nvec, device=device, **kwargs)
