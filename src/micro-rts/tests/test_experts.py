"""Thin integration tests: the built-in expert AIs load, run, and step.

Each expert is wired in as the player-2 opponent of its own parallel sub-env
(see ``conftest.env``). Stepping the shared env therefore drives every expert's
Java-side logic at once; we assert the games advance without error.
"""

import numpy as np

from conftest import EXPERT_NAMES, noop_action


def test_all_experts_resolve(experts):
    """Every named expert resolves to a callable factory in microrts_ai."""
    assert len(experts) == len(EXPERT_NAMES)
    assert all(callable(ai) for ai in experts)


def test_one_subenv_per_expert(env, experts):
    assert env.num_envs == len(experts)


def test_experts_run_and_step(env):
    """All experts step their games forward over several frames."""
    obs0 = np.asarray(env.reset())
    action = noop_action(env.num_envs)

    advanced = np.zeros(env.num_envs, dtype=bool)
    last_obs = obs0
    for _ in range(20):
        obs, reward, done, info = env.step(action)
        obs = np.asarray(obs)
        assert obs.shape == obs0.shape
        # Track which experts have changed the board state at least once; the
        # active rush/coac/MCTS bots move units, so their planes must change.
        advanced |= np.any(obs != last_obs, axis=(1, 2, 3))
        last_obs = obs

    # passiveAI never acts, but every other expert should perturb the board.
    active = [i for i, name in enumerate(EXPERT_NAMES) if name != "passiveAI"]
    assert advanced[active].all(), "some active expert never moved a unit"
