"""Training-health helpers: collapse diagnostics shared by RL trainers.

Pure functions (no state) so they are trivial to unit-test and reuse. Stateful
policy reactions (early-stop counters, best tracking) live in the trainers.
"""

from __future__ import annotations

import math

import torch


def is_finite(value) -> bool:
    """True if a scalar / tensor / dict-of-scalars contains only finite numbers."""
    if isinstance(value, dict):
        return all(is_finite(v) for v in value.values())
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """Fraction of the return variance explained by the value function.

    ``1`` is perfect, ``0`` means no better than predicting the mean, and negative
    means actively worse — a classic sign the critic is diverging / collapsing.
    """
    values, returns = values.flatten().float(), returns.flatten().float()
    var_returns = returns.var(unbiased=False)
    if var_returns == 0:
        return float("nan")
    return float(1 - (returns - values).var(unbiased=False) / var_returns)


def max_entropy(action_nvec: torch.Tensor) -> float:
    """Maximum achievable entropy for a MultiDiscrete action (uniform policy)."""
    return float(torch.log(action_nvec.float()).sum())


def entropy_fraction(entropy: float, action_nvec: torch.Tensor) -> float:
    """Current entropy as a fraction of the uniform-policy maximum (in [0, 1])."""
    m = max_entropy(action_nvec)
    return entropy / m if m > 0 else float("nan")
