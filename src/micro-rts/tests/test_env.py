"""Thin integration tests: the MicroRTS env builds, resets, and steps."""

import numpy as np

from conftest import (
    ACTION_NVEC,
    MAP_HEIGHT,
    MAP_WIDTH,
    OBS_PLANES,
    noop_action,
)


def test_observation_space(env):
    assert tuple(env.observation_space.shape) == (MAP_HEIGHT, MAP_WIDTH, OBS_PLANES)


def test_action_space(env):
    assert env.action_space.nvec.tolist() == ACTION_NVEC


def test_reset_shape(env):
    obs = np.asarray(env.reset())
    assert obs.shape == (env.num_envs, MAP_HEIGHT, MAP_WIDTH, OBS_PLANES)
    # One-hot encoded planes are strictly binary.
    assert set(np.unique(obs).tolist()) <= {0, 1}


def test_step_returns_consistent_shapes(env):
    env.reset()
    obs, reward, done, info = env.step(noop_action(env.num_envs))
    obs = np.asarray(obs)
    assert obs.shape == (env.num_envs, MAP_HEIGHT, MAP_WIDTH, OBS_PLANES)
    assert np.asarray(reward).shape == (env.num_envs,)
    assert np.asarray(done).shape == (env.num_envs,)
    assert len(info) == env.num_envs
    assert "raw_rewards" in info[0]
    # Six raw reward components feed the weighted scalar reward.
    assert np.asarray(info[0]["raw_rewards"]).shape == (6,)


def test_random_action_steps(env):
    """A randomly sampled (in-bounds) action plan steps without error."""
    env.reset()
    rng = np.random.default_rng(0)
    actions = np.stack(
        [rng.integers(low=0, high=ACTION_NVEC) for _ in range(env.num_envs)]
    )[:, None, :].astype(np.int32)
    obs, reward, done, info = env.step(actions)
    assert np.asarray(obs).shape == (env.num_envs, MAP_HEIGHT, MAP_WIDTH, OBS_PLANES)
