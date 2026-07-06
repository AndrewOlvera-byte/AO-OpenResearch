"""``ResNetEncoder`` — turns a MicroRTS obs grid into a flat feature vector.

Wraps the pre-activation ``ResNetBackbone`` (``models/shared/ResNet.py``) in
feature-extraction mode (``num_classes=None``), takes the deepest stage
(``res5``), global-average-pools it, and flattens to ``[B, out_dim]``.

Defaults are compact and tuned for the 16x16 ``basesWorkers`` map: basic
pre-activation blocks, four stages, modest widths. The stem (stride-2 conv +
stride-2 maxpool) downsamples 16 -> 4 before the stages, so even with stage
strides the spatial map never collapses below 1x1.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from .ResNet import BasicBlockPreAct, ResNetBackbone, ResNetConfig


def _group_norm(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm whose group count is clamped to a divisor of ``num_channels``.

    GroupNorm is batch-independent, so the encoder behaves identically during
    rollout collection (``policy.step``) and the PPO update (``evaluate_actions``).
    BatchNorm would instead use/refresh running stats over the rollout batch and
    then minibatch stats during the update — a train/collect mismatch that is a
    well-known source of PPO instability. Hence GroupNorm is the default here.
    """
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int = 32,
        layers: tuple[int, int, int, int] = (2, 2, 2, 2),
        stage_planes: tuple[int, int, int, int] = (32, 64, 128, 256),
        stem_type: str = "classic",
        drop_path_rate: float = 0.0,
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        config = ResNetConfig(
            in_channels=in_channels,
            stem_channels=stem_channels,
            layers=layers,
            stage_planes=stage_planes,
            num_classes=None,
            stem_type=stem_type,
            drop_path_rate=drop_path_rate,
        )
        norm_layer = partial(_group_norm, num_groups=norm_groups)
        self.backbone = ResNetBackbone(block=BasicBlockPreAct, config=config, norm_layer=norm_layer)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = stage_planes[-1] * BasicBlockPreAct.expansion

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        pooled = self.global_pool(features["res5"])
        return torch.flatten(pooled, 1)
