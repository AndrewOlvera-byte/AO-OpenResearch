"""Patched-jar opponent-action + seat-swap integration tests.

The world-model v3 data contract (NEXT_PLAN.md, Workstream A) needs two things
only the patched microrts.jar provides:

1. ``opponent_action`` — the scripted bot's per-tick gridnet action, exposed by
   the env alongside the learner's own submitted action.
2. ``player_ids`` — per-lane seat of the Python player, so the scripted bot can
   occupy the player-0 role (role-swapped collection).

The crucial correctness property is **alignment**: the exposed opponent action
must be the one issued in the *same* engine cycle as the learner's submitted
action (the pair that produced the returned obs). We verify it end-to-end by
replay: record a bot game (deterministic workerRushAI, learner NOOPs), then
re-run the recorded action streams through a *self-play* env — where both
players are Python-fed — from the same initial state. Faithful encoding + tick
alignment ⇒ identical observation trajectories.

Skipped (not failed) on an unpatched jar, so the suite still passes on stock
gym-microrts 0.3.2.

NOTE: env.close() shuts down the whole JVM (gym_microrts calls shutdownJVM),
which cannot restart in-process — so tests that need a second env never close
the first; the --forked per-test process reclaims everything on exit.
"""

import numpy as np
import pytest
import torch

from conftest import MAP_HEIGHT, MAP_WIDTH

CELLS = MAP_HEIGHT * MAP_WIDTH
N_COMP = 7  # [action_type, move, harvest, return, produce_dir, produce_type, attack]
CELL_NVEC = (6, 4, 4, 4, 4, 7, 49)
OWNER_OWN, OWNER_ENEMY = 11, 12  # obs channels: hp(5)+res(5)+owner(none,own,enemy)


def _make_env(**overrides):
    from environments.microrts_env import EnvConfig, MicroRTSVecEnv

    kwargs = dict(num_envs=1, max_steps=500, mode="bot", bots=("workerRushAI",),
                  gridnet=True, opponent_action=True)
    kwargs.update(overrides)
    try:
        return MicroRTSVecEnv(EnvConfig(**kwargs))
    except AssertionError as e:
        if "patched jar" in str(e):
            pytest.skip("needs the patched microrts.jar (infra/microrts-jar-patch)")
        raise


def _noop(n):
    return torch.zeros(n, CELLS, N_COMP, dtype=torch.long)


def test_opponent_action_shape_and_bounds():
    env = _make_env()
    env.async_reset()
    td = env.recv()
    opp = td["opponent_action"]
    assert opp.shape == (1, CELLS, N_COMP)
    assert (opp == 0).all()  # no tick has happened yet

    saw_nonzero = False
    for _ in range(16):
        td = env.step(_noop(1))
        opp = td["opponent_action"]
        assert opp.shape == (1, CELLS, N_COMP)
        for c, hi in enumerate(CELL_NVEC):
            assert opp[..., c].min() >= 0 and opp[..., c].max() < hi
        saw_nonzero |= bool((opp != 0).any())
    # workerRushAI moves/harvests within the first few ticks.
    assert saw_nonzero, "opponent never acted in 16 steps — encoding broken?"
    env.close()


def test_opponent_action_alignment_replay():
    """Recorded (learner, opponent) streams replayed through a self-play env
    reproduce the bot game's observation trajectory tick for tick."""
    steps = 48
    bot = _make_env()
    bot.async_reset()
    td = bot.recv()
    obs_stream, opp_stream = [td["obs"].clone()], []
    for _ in range(steps):
        td = bot.step(_noop(1))
        obs_stream.append(td["obs"].clone())
        opp_stream.append(td["opponent_action"].clone())
        if bool(td["done"].any()):
            pytest.fail("episode ended inside the replay window; shorten steps")

    from environments.microrts_env import EnvConfig, MicroRTSVecEnv

    sp = MicroRTSVecEnv(EnvConfig(num_envs=2, max_steps=500, mode="selfplay",
                                  gridnet=True))
    sp.async_reset()
    td = sp.recv()
    # Same map, same engine: identical initial frame for player 0.
    assert torch.equal(td["obs"][0], obs_stream[0][0])
    for t in range(steps):
        actions = torch.zeros(2, CELLS, N_COMP, dtype=torch.long)
        actions[1] = opp_stream[t][0]           # player 1 <- recorded bot action
        td = sp.step(actions)
        assert torch.equal(td["obs"][0], obs_stream[t + 1][0]), \
            f"replay diverged at tick {t}: opponent action mis-encoded/mis-aligned"


def test_player_ids_seat_swap():
    """player_ids=(1,) seats the Python player as player 1: its own units sit
    where a seat-0 env sees the enemy, and the engine mask follows the seat."""
    a = _make_env()                       # python = player 0 (stock)
    td_a = a.reset()
    obs_a, mask_a = td_a["obs"][0], td_a["mask"][0]

    b = _make_env(player_ids=(1,))        # python = player 1, bot = player 0
    td_b = b.reset()
    obs_b, mask_b = td_b["obs"][0], td_b["mask"][0]

    own_b = obs_b[OWNER_OWN].flatten().bool()
    enemy_a = obs_a[OWNER_ENEMY].flatten().bool()
    own_a = obs_a[OWNER_OWN].flatten().bool()
    assert torch.equal(own_b, enemy_a), "seat-1 obs is not the mirrored perspective"

    sel_a = mask_a[:, 0].bool()           # mask channel 0: source-selectable cells
    sel_b = mask_b[:, 0].bool()
    assert sel_a.any() and sel_b.any()
    assert not torch.equal(sel_a, sel_b), "mask did not follow the seat swap"
    assert (sel_b & own_b).sum() == sel_b.sum(), "selectable cells not on own units"

    # The scripted bot now plays player 0 (the 'player 1' seat in 1-indexed
    # terms): its exposed actions land on cells the seat-0 run owned itself.
    acted = torch.zeros(CELLS, dtype=torch.bool)
    for _ in range(16):
        td_b = b.step(_noop(1))
        acted |= (td_b["opponent_action"][0].abs().sum(-1) > 0)
    assert acted.any()
    assert (acted & ~own_a).sum() == 0, \
        "role-swapped bot acted outside the player-0 start area at game start"
