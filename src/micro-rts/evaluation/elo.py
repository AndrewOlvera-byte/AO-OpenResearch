"""Elo estimation from match results against fixed-rating opponents.

We don't run a full round-robin between agents; instead each scripted bot is an
*anchor* with a fixed rating, and we back out the agent's implied rating from its
score against that bot via the standard logistic Elo model:

    expected_score(R_a, R_b) = 1 / (1 + 10 ** ((R_b - R_a) / 400))

Inverting for the agent given a realized score ``S`` in [0, 1] against a bot at
rating ``R_bot``:

    R_agent = R_bot + 400 * log10(S / (1 - S))

A perfect (S=1) or winless (S=0) record would send that to +/-inf, so ``S`` is
clamped to a half-win band based on the number of games (Laplace-style), which
also encodes "more games -> more confident extreme".
"""

from __future__ import annotations

import math

DEFAULT_ANCHOR = 1000.0


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def score(wins: int, draws: int, games: int) -> float:
    """Match score in [0, 1]: win=1, draw=0.5, loss=0."""
    if games <= 0:
        return float("nan")
    return (wins + 0.5 * draws) / games


def implied_rating(wins: int, draws: int, games: int, opponent_rating: float = DEFAULT_ANCHOR) -> float:
    """Agent rating implied by its record vs a fixed-rating opponent."""
    if games <= 0:
        return float("nan")
    s = score(wins, draws, games)
    # Clamp to a half-game band so 0% / 100% stay finite (tighter with more games).
    lo = 1.0 / (2.0 * games)
    s = min(max(s, lo), 1.0 - lo)
    return opponent_rating + 400.0 * math.log10(s / (1.0 - s))


def aggregate_rating(per_bot: dict[str, dict], anchors: dict[str, float] | None = None) -> float:
    """Games-weighted mean of the per-bot implied ratings."""
    anchors = anchors or {}
    num, den = 0.0, 0
    for bot, rec in per_bot.items():
        g = rec["games"]
        if g <= 0:
            continue
        r = implied_rating(rec["wins"], rec["draws"], g, anchors.get(bot, DEFAULT_ANCHOR))
        num += r * g
        den += g
    return num / den if den else float("nan")
