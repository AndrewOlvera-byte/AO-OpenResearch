"""Multiprocess EnvPool: W workers, each its own JVM, aggregated behind VecEnv.

The test process itself never builds an env (so it stays JVM-free); the pool spawns
worker processes that each own a JVM. Same VecEnv contract as MicroRTSVecEnv, so the
existing Collector works unchanged.
"""

import torch

from collectors.collector import Collector
from models.random_policy import RandomPolicy


def _cfg(num_envs):
    from environments.microrts_env import EnvConfig

    return EnvConfig(num_envs=num_envs, max_steps=200, mode="bot",
                     bots=("randomBiasedAI", "workerRushAI"))


def test_pool_spaces_and_reset():
    from collectors.vecpool import MultiprocessPool

    pool = MultiprocessPool(_cfg(4), num_workers=2)
    try:
        assert pool.num_envs == 4
        assert pool.obs_shape == (27, 16, 16)
        assert pool.action_nvec.tolist() == [256, 6, 4, 4, 4, 4, 7, 49]
        trans = pool.reset()
        assert tuple(trans["obs"].shape) == (4, 27, 16, 16)
    finally:
        pool.close()


def test_pool_step_and_aggregation():
    from collectors.vecpool import MultiprocessPool

    pool = MultiprocessPool(_cfg(6), num_workers=3)
    try:
        policy = RandomPolicy(pool.action_nvec)
        trans = pool.reset()
        out = policy.step(trans["obs"])
        nxt = pool.step(out["action"])
        assert tuple(nxt["obs"].shape) == (6, 27, 16, 16)
        assert nxt["reward"].shape == (6,)
        # env_id spans all aggregated envs across workers.
        assert sorted(nxt["env_id"].tolist()) == list(range(6))
    finally:
        pool.close()


def test_collector_runs_on_pool():
    from collectors.vecpool import MultiprocessPool

    pool = MultiprocessPool(_cfg(4), num_workers=2)
    try:
        policy = RandomPolicy(pool.action_nvec)
        buf = Collector(pool, policy, horizon=8).collect()
        assert tuple(buf.data["obs"].shape) == (8, 4, 27, 16, 16)
        assert torch.isfinite(buf.data["advantage"]).all()
    finally:
        pool.close()
