"""Reward shaping presets for MicroRTS.

gym_microrts exposes 6 raw reward components weighted into a scalar:
``(win, resource, worker, building, attack, combat-unit)``. Centralizing the
weight vectors here lets experiment configs pick a named preset instead of
hard-coding magic numbers in the env.
"""

from __future__ import annotations

# (win, resource, worker, building, attack, combat-unit)
REWARD_PRESETS: dict[str, tuple[float, ...]] = {
    # Balanced dense shaping — good for bootstrapping a from-scratch policy.
    "dense_shaped": (10.0, 1.0, 1.0, 0.2, 1.0, 4.0),
    # Mostly the win/loss signal, light resource nudge to avoid total sparsity.
    "sparse_winloss": (10.0, 0.1, 0.0, 0.0, 0.0, 0.0),
    # Win-dominated target: the real objective is winning, not farming shaped
    # reward. A tiny resource term keeps a gradient before the agent can win.
    "win_focused": (10.0, 0.05, 0.0, 0.0, 0.0, 0.0),
    # Pure game outcome: +1 win / -1 loss / 0 draw. Used by evaluation.
    "win_loss": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}

# Back-compat alias for the const previously inlined in microrts_env.py.
DEFAULT_REWARD_WEIGHT = REWARD_PRESETS["dense_shaped"]


def reward_weight(preset: str = "dense_shaped") -> tuple[float, ...]:
    if preset not in REWARD_PRESETS:
        raise KeyError(f"unknown reward preset {preset!r}; choices: {list(REWARD_PRESETS)}")
    return REWARD_PRESETS[preset]


def lerp_weights(start: tuple[float, ...], end: tuple[float, ...], t: float) -> tuple[float, ...]:
    """Linear interpolation between two weight vectors; ``t`` clamped to [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    return tuple(s + (e - s) * t for s, e in zip(start, end))
