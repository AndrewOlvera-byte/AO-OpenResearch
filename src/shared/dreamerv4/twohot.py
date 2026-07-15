"""Symlog two-hot regression — the DreamerV3/4 stabilizer for scalar heads.

MSE on sparse ±1 rewards (or wide-range returns) is minimized by predicting the
mean, which for Atari is ≈0 everywhere: the rare nonzero events that are the whole
point contribute almost nothing to the loss, so the reward/value head learns a
constant. DreamerV3 fixes this by casting scalar regression as *classification* over
a fixed grid of exponentially-spaced bins in **symlog** space, trained with a
two-hot cross-entropy. Symlog (``sign(x)·log(1+|x|)``) compresses large magnitudes
so a single bin grid covers both ±1 rewards and large returns; the two-hot target
puts a rare event on its own bin, so it can no longer be washed out by the zeros.

This module is deliberately env-agnostic (a ``TwoHot`` coder owning only the bin
grid, plus the ``symlog``/``symexp`` transforms) so the Atari and MicroRTS Dreamer
builds share one implementation for both the reward head and the critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x)·log(1+|x|)`` — a signed, magnitude-compressing transform."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`symlog`: ``sign(x)·(exp(|x|)-1)``."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Two-hot encode scalars ``x`` (...) onto ``bins`` (K,) -> (..., K).

    ``x`` is expected in the same (symlog) space as ``bins``. Each value lands
    between two adjacent bins with weights that linearly interpolate its position,
    so ``(two_hot(x) * bins).sum(-1) == x`` for ``x`` inside the grid.
    """
    x = x.clamp(min=float(bins[0]), max=float(bins[-1]))
    below = (bins <= x[..., None]).sum(-1) - 1          # index of bin at/below x
    below = below.clamp(0, bins.numel() - 1)
    above = (below + 1).clamp(0, bins.numel() - 1)
    bin_below, bin_above = bins[below], bins[above]
    span = bin_above - bin_below
    w_below = torch.where(span.abs() < 1e-8, torch.ones_like(span), (bin_above - x) / span)
    w_above = 1.0 - w_below
    K = bins.numel()
    return F.one_hot(below, K).float() * w_below[..., None] \
        + F.one_hot(above, K).float() * w_above[..., None]


class TwoHot(nn.Module):
    """Coder over a fixed symlog bin grid: decode logits -> scalar, and its loss.

    Holds no projection of its own — the caller's head/MLP emits the ``K`` logits.
    ``low``/``high`` are in symlog space; the default ``[-20, 20]`` covers roughly
    ``±5e8`` in real units, so it fits ±1 rewards and any realistic return.
    """

    def __init__(self, num_bins: int = 255, low: float = -20.0, high: float = 20.0) -> None:
        super().__init__()
        self.num_bins = int(num_bins)
        self.register_buffer("bins", torch.linspace(low, high, self.num_bins))

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """Expected value under ``softmax(logits)``, mapped back to real space."""
        probs = F.softmax(logits, dim=-1)
        return symexp((probs * self.bins).sum(-1))

    def loss(self, logits: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
        """Two-hot cross-entropy of ``logits`` against real-valued ``target``.

        ``mask`` (same shape as ``target``, bool/float) selects which elements
        contribute; the loss is the mean over selected elements only.
        """
        tgt = two_hot(symlog(target), self.bins).detach()
        log_probs = F.log_softmax(logits, dim=-1)
        ce = -(tgt * log_probs).sum(-1)
        if mask is None:
            return ce.mean()
        m = mask.to(ce.dtype)
        return (ce * m).sum() / m.sum().clamp_min(1.0)
