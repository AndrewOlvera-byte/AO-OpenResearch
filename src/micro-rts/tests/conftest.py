"""Shared fixtures for the micro-rts integration tests.

JPype starts a single JVM per process and *cannot restart it* once shut down.
The suite runs under ``--forked`` (see pyproject), so each test gets its own
process and therefore its own clean JVM — letting independent tests each build
their own env (bot, self-play, collector, ...) without colliding.
"""

import numpy as np
import pytest

from gym_microrts import microrts_ai

# Built-in scripted experts shipped with MicroRTS that we smoke-test. Each one
# becomes the player-2 opponent of a parallel sub-env.
EXPERT_NAMES = [
    "passiveAI",
    "workerRushAI",
    "lightRushAI",
    "coacAI",
    "randomBiasedAI",
    "naiveMCTSAI",
]

MAP_PATH = "maps/16x16/basesWorkers16x16.xml"
MAP_HEIGHT = 16
MAP_WIDTH = 16
OBS_PLANES = 27  # hp(5) + resources(5) + owner(3) + unit_type(8) + action(6)
ACTION_NVEC = [256, 6, 4, 4, 4, 4, 7, 49]
REWARD_WEIGHT = np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0])


@pytest.fixture(scope="session")
def experts():
    return [getattr(microrts_ai, name) for name in EXPERT_NAMES]


@pytest.fixture(scope="session")
def env(experts):
    """One shared vec-env: one parallel sub-env per expert opponent."""
    from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=len(experts),
        max_steps=300,
        ai2s=experts,
        map_path=MAP_PATH,
        reward_weight=REWARD_WEIGHT,
    )
    yield env
    env.close()


def noop_action(num_envs):
    """A valid no-op action batch: ``int[][][]`` of shape (num_envs, 1, 8)."""
    return np.zeros((num_envs, 1, len(ACTION_NVEC)), dtype=np.int32)


@pytest.fixture
def bot_env():
    """MicroRTSVecEnv wrapper, bot mode (experts cycled across envs)."""
    from environments.microrts_env import EnvConfig, MicroRTSVecEnv

    env = MicroRTSVecEnv(
        EnvConfig(num_envs=4, max_steps=200, mode="bot",
                  bots=("randomBiasedAI", "workerRushAI"))
    )
    yield env
    env.close()


@pytest.fixture
def selfplay_env():
    """MicroRTSVecEnv wrapper, self-play mode (agent controls both players)."""
    from environments.microrts_env import EnvConfig, MicroRTSVecEnv

    env = MicroRTSVecEnv(EnvConfig(num_envs=4, max_steps=200, mode="selfplay"))
    yield env
    env.close()
