"""
Modern pre-activation ResNet backbone in PyTorch.

Design goals:
- clean residual gradient flow
- efficient bottleneck blocks
- stable initialization
- readable research code
- easy extension for detection / segmentation / classification

Recommended defaults:
    model = ResNetBackbone(
        block=BottleneckPreAct,
        layers=(3, 4, 6, 3),      # ResNet-50
        stem_type="deep",
        num_classes=1000,
        drop_path_rate=0.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------
# Utility layers
# ---------------------------------------------------------------------

def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 3,
        stride = stride,
        padding = dilation,
        groups = groups,
        bias = False,
        dilation = dilation,
    )

def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,

) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size = 1,
        stride = stride,
        bias = False
    )

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks), more effective than dropout inside convnets."""
    def __init__(self, drop_prob: float = 0.0) ->None:
        super().__init__()
        if drop_prob < 0.0 or drop_prob > 1.0:
            raise ValueError(f"drop_prob should be in range [0.0, 1.0], but got {drop_prob}")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)

        return x.div(keep_prob) * mask
    
# ---------------------------------------------------------------------
# Residual blocks
# ---------------------------------------------------------------------

class BottleneckPreAct(nn.Module):
    """ Pre-activation bottleneck ResNet block. """
    expansion: int = 4

    def __init__(
        self, 
        in_channels: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0
    ) -> None:
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        out_channels = planes * self.expansion

        self.bn1 = norm_layer(in_channels)
        self.conv1 = conv1x1(in_channels, width)

        self.bn2 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)

        self.bn3 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.proj: Optional[nn.Conv2d]
        if stride !=1 or in_channels != out_channels:
            self.proj = conv1x1(in_channels, out_channels, stride=stride)
        else:
            self.proj = None
        
    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(x))

        if self.proj is not None:
            shortcut = self.proj(out)
        else:
            shortcut = x
        
        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.relu(self.bn3(out))
        out = self.conv3(out)

        out = self.drop_path(out)

        return shortcut + out
class BasicBlockPreAct(nn.Module):
    """
    Pre-activation basic residual block.

    Better for smaller models such as ResNet-18 / ResNet-34.
    Less compute-efficient than bottlenecks at large width/depth.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlockPreAct only supports groups=1 and base_width=64.")

        out_channels = planes * self.expansion

        self.bn1 = norm_layer(in_channels)
        self.conv1 = conv3x3(in_channels, planes, stride=stride, dilation=dilation)

        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, out_channels, dilation=dilation)

        self.relu = nn.ReLU(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.proj: Optional[nn.Conv2d]
        if stride != 1 or in_channels != out_channels:
            self.proj = conv1x1(in_channels, out_channels, stride=stride)
        else:
            self.proj = None

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(x))

        if self.proj is not None:
            shortcut = self.proj(out)
        else:
            shortcut = x

        out = self.conv1(out)

        out = self.relu(self.bn2(out))
        out = self.conv2(out)

        out = self.drop_path(out)

        return shortcut + out


# ---------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------


class DeepStem(nn.Module):
    """
    Modern ResNet stem.

    Instead of:
        7x7 conv stride 2

    use:
        3x3 conv stride 2
        3x3 conv stride 1
        3x3 conv stride 1

    This tends to preserve local detail better and is commonly used in
    stronger ResNet variants.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        mid_channels = out_channels // 2

        self.net = nn.Sequential(
            conv3x3(in_channels, mid_channels, stride=2),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            conv3x3(mid_channels, mid_channels, stride=1),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
            conv3x3(mid_channels, out_channels, stride=1),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ClassicStem(nn.Module):
    """
    Original ResNet-style stem.

        7x7 conv stride 2
        BN
        ReLU
        maxpool

    Still useful for compatibility with old checkpoints.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------
# Backbone config
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class ResNetConfig:
    """
    Configuration for ResNetBackbone.
    """

    in_channels: int = 3
    stem_channels: int = 64
    layers: tuple[int, int, int, int] = (3, 4, 6, 3)
    stage_planes: tuple[int, int, int, int] = (64, 128, 256, 512)
    num_classes: Optional[int] = 1000

    stem_type: str = "deep"  # "deep" or "classic"

    groups: int = 1
    base_width: int = 64

    replace_stride_with_dilation: tuple[bool, bool, bool] = (False, False, False)

    drop_path_rate: float = 0.0
    zero_init_residual: bool = True


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------


class ResNetBackbone(nn.Module):
    """
    Pre-activation ResNet backbone.

    Returns:
    - if num_classes is not None:
        classification logits

    - if return_features=True:
        dictionary of intermediate feature maps:
            {
                "stem": Tensor,
                "res2": Tensor,
                "res3": Tensor,
                "res4": Tensor,
                "res5": Tensor,
            }

    Spatial downsampling for default deep stem:
        input: H x W
        stem: H/4 x W/4 after maxpool
        res2: H/4
        res3: H/8
        res4: H/16
        res5: H/32

    With dilation replacement, later stages can preserve higher resolution.
    """

    def __init__(
        self,
        block: type[nn.Module] = BottleneckPreAct,
        config: ResNetConfig = ResNetConfig(),
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()

        self.config = config
        self.norm_layer = norm_layer
        self.inplanes = config.stem_channels
        self.dilation = 1

        if config.stem_type == "deep":
            self.stem = DeepStem(
                in_channels=config.in_channels,
                out_channels=config.stem_channels,
                norm_layer=norm_layer,
            )
        elif config.stem_type == "classic":
            self.stem = ClassicStem(
                in_channels=config.in_channels,
                out_channels=config.stem_channels,
                norm_layer=norm_layer,
            )
        else:
            raise ValueError(f"Unknown stem_type: {config.stem_type}")

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        total_blocks = sum(config.layers)
        drop_rates = torch.linspace(0, config.drop_path_rate, total_blocks).tolist()
        drop_iter = iter(drop_rates)

        self.res2 = self._make_stage(
            block=block,
            planes=config.stage_planes[0],
            blocks=config.layers[0],
            stride=1,
            dilate=False,
            drop_iter=drop_iter,
        )

        self.res3 = self._make_stage(
            block=block,
            planes=config.stage_planes[1],
            blocks=config.layers[1],
            stride=2,
            dilate=config.replace_stride_with_dilation[0],
            drop_iter=drop_iter,
        )

        self.res4 = self._make_stage(
            block=block,
            planes=config.stage_planes[2],
            blocks=config.layers[2],
            stride=2,
            dilate=config.replace_stride_with_dilation[1],
            drop_iter=drop_iter,
        )

        self.res5 = self._make_stage(
            block=block,
            planes=config.stage_planes[3],
            blocks=config.layers[3],
            stride=2,
            dilate=config.replace_stride_with_dilation[2],
            drop_iter=drop_iter,
        )

        final_channels = config.stage_planes[-1] * block.expansion

        # Final activation is useful in pre-activation networks because blocks
        # themselves do not apply ReLU after residual addition.
        self.final_bn = norm_layer(final_channels)
        self.final_relu = nn.ReLU(inplace=True)

        if config.num_classes is not None:
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(final_channels, config.num_classes)
        else:
            self.global_pool = None
            self.fc = None

        self._init_weights(zero_init_residual=config.zero_init_residual)

    def _make_stage(
        self,
        block: type[nn.Module],
        planes: int,
        blocks: int,
        stride: int,
        dilate: bool,
        drop_iter,
    ) -> nn.Sequential:
        """
        Construct one ResNet stage.

        The first block may downsample spatially or increase dilation.
        Remaining blocks keep the same shape.
        """

        previous_dilation = self.dilation

        if dilate:
            self.dilation *= stride
            stride = 1

        layers: list[nn.Module] = []

        layers.append(
            block(
                in_channels=self.inplanes,
                planes=planes,
                stride=stride,
                groups=self.config.groups,
                base_width=self.config.base_width,
                dilation=previous_dilation,
                norm_layer=self.norm_layer,
                drop_path=next(drop_iter),
            )
        )

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.inplanes,
                    planes=planes,
                    stride=1,
                    groups=self.config.groups,
                    base_width=self.config.base_width,
                    dilation=self.dilation,
                    norm_layer=self.norm_layer,
                    drop_path=next(drop_iter),
                )
            )

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool = True) -> None:
        """
        Kaiming initialization for convs.

        For stability:
        - BatchNorm gamma starts as 1
        - BatchNorm beta starts as 0
        - final BN in each residual branch can be zero-initialized

        Zero-init residual means each residual branch starts near zero,
        so every block initially behaves close to identity.
        """

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, BottleneckPreAct):
                    nn.init.zeros_(module.bn3.weight)
                elif isinstance(module, BasicBlockPreAct):
                    nn.init.zeros_(module.bn2.weight)

    def forward_features(self, x: Tensor) -> dict[str, Tensor]:
        """
        Return multiscale features.

        Useful for:
        - FPN
        - segmentation heads
        - contrastive learning
        - dense prediction
        """

        x = self.stem(x)
        x = self.maxpool(x)
        stem = x

        res2 = self.res2(stem)
        res3 = self.res3(res2)
        res4 = self.res4(res3)
        res5 = self.res5(res4)

        res5 = self.final_relu(self.final_bn(res5))

        return {
            "stem": stem,
            "res2": res2,
            "res3": res3,
            "res4": res4,
            "res5": res5,
        }

    def forward(self, x: Tensor, return_features: bool = False) -> Tensor | dict[str, Tensor]:
        features = self.forward_features(x)

        if return_features or self.fc is None:
            return features

        x = features["res5"]
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
