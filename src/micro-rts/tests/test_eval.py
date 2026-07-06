"""Elo estimation + eval report aggregation (pure, no JVM)."""

import math

from evaluation import elo


def test_expected_score_symmetry():
    assert abs(elo.expected_score(1000, 1000) - 0.5) < 1e-9
    a = elo.expected_score(1200, 1000)
    b = elo.expected_score(1000, 1200)
    assert abs((a + b) - 1.0) < 1e-9 and a > 0.5


def test_score_and_implied_rating_monotonic():
    assert elo.score(10, 0, 20) == 0.5
    r_even = elo.implied_rating(10, 0, 20, 1000)
    assert abs(r_even - 1000) < 1e-6  # 50% vs a 1000 bot -> ~1000
    r_strong = elo.implied_rating(18, 0, 20, 1000)
    r_weak = elo.implied_rating(2, 0, 20, 1000)
    assert r_strong > 1000 > r_weak


def test_implied_rating_clamps_extremes():
    # 100% win rate stays finite (clamped by the half-game band).
    r = elo.implied_rating(20, 0, 20, 1000)
    assert math.isfinite(r) and r > 1000


def test_aggregate_rating_games_weighted():
    per_bot = {
        "weak": {"wins": 18, "draws": 0, "losses": 2, "games": 20},
        "strong": {"wins": 2, "draws": 0, "losses": 18, "games": 20},
    }
    agg = elo.aggregate_rating(per_bot)
    assert abs(agg - 1000) < 50  # roughly balanced overall
