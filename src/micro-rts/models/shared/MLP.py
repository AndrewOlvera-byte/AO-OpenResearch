"""Small MLP helpers shared by policy necks and heads.

Keeps the actual ``nn`` plumbing (layer sizing, activation, optional LayerNorm,
orthogonal init) in one place so policies stay declarative.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


def orthogonal_init(module: nn.Module, gain: float = nn.init.calculate_gain("relu")) -> nn.Module:
    """Orthogonal weight init with zero bias — the PPO/A2C default.

    Apply with a small ``gain`` (e.g. 0.01) on the policy logits layer to start
    near-uniform, and ``gain=1.0`` on the value head.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    ``hidden_dims`` is the list of hidden widths; a final ``Linear(*, out_dim)``
    is appended with no activation so the caller decides what to do with the
    output (logits, value, neck features).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        layernorm: bool = False,
        out_gain: float | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(activation())
            prev = h
        self.out = nn.Linear(prev, out_dim)
        self.body = nn.Sequential(*layers)
        self.out_dim = out_dim

        orthogonal_init(self.body)
        if out_gain is not None:
            orthogonal_init(self.out, gain=out_gain)
        else:
            orthogonal_init(self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))
