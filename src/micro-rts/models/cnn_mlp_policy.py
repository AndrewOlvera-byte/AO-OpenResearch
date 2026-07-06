"""``CNNMLPPolicy`` — ResNet obs-encoder -> MLP neck -> MLP actor + critic heads.

The real research policy: a shared pre-activation ResNet encoder feeds a small MLP
neck, which forks into a MultiDiscrete actor head and a scalar critic head. The
shared encoder means actor and critic see the same representation, which is what we
want when learning the encoder from scratch under PPO.

Satisfies the ``Policy`` protocol (``environments/base.py``): the API mirrors
``models/cnn_policy.py`` exactly (``step`` / ``evaluate_actions``) so it drops into the
``Collector`` and ``PPOTrainer`` unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from core.registry import register

from .shared.ActionHead import MultiDiscreteActionHead
from .shared.Encoder import ResNetEncoder
from .shared.MLP import MLP


class CNNMLPPolicy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_nvec,
        device: str = "cpu",
        encoder: dict | None = None,
        neck_hidden: tuple[int, ...] = (256,),
        neck_out: int = 256,
        critic_hidden: tuple[int, ...] = (256,),
    ) -> None:
        super().__init__()
        c, _, _ = obs_shape
        if not torch.is_tensor(action_nvec):
            action_nvec = torch.as_tensor(list(action_nvec), dtype=torch.long)

        self.encoder = ResNetEncoder(in_channels=c, **(encoder or {}))
        self.neck = MLP(self.encoder.out_dim, list(neck_hidden), neck_out, layernorm=True)
        self.actor = MultiDiscreteActionHead(neck_out, action_nvec)
        self.critic = MLP(neck_out, list(critic_hidden), 1, out_gain=1.0)

        self.device = device
        self.to(device)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.neck(self.encoder(obs))

    @torch.no_grad()
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None,
             deterministic: bool = False) -> TensorDict:
        feat = self._features(obs.to(self.device))
        action, logprob = self.actor.sample(feat, deterministic=deterministic)
        return TensorDict(
            {
                "action": action,
                "logprob": logprob,
                "value": self.critic(feat).squeeze(-1),
            },
            batch_size=[obs.shape[0]],
        )

    def evaluate_actions(self, obs: torch.Tensor, action: torch.Tensor, mask=None):
        """For the PPO update: new logprob, entropy, value of stored actions.

        ``mask`` is accepted for a uniform Policy API but ignored — this head does
        no invalid-action masking (see ``MaskedActorPolicy`` for the masked variant).
        """
        feat = self._features(obs)
        logprob, entropy = self.actor.log_prob_entropy(feat, action)
        return logprob, entropy, self.critic(feat).squeeze(-1)

    def freeze_encoder(self, frozen: bool = True) -> None:
        """Toggle encoder grads — the curriculum freezes it early to stabilize."""
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)

    def unfreeze_encoder(self) -> None:
        self.freeze_encoder(False)


@register("model", "cnn_mlp")
def build_cnn_mlp(obs_shape, action_nvec, device="cpu", **kwargs) -> CNNMLPPolicy:
    return CNNMLPPolicy(obs_shape, action_nvec, device=device, **kwargs)
