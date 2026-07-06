"""Integration: MicroRTSVecEnv adapter + RandomPolicy + Collector end-to-end."""

import torch

from collectors.collector import Collector
from models.random_policy import RandomPolicy


def test_bot_env_reset_and_spaces(bot_env):
    assert bot_env.num_envs == 4
    assert bot_env.obs_shape == (27, 16, 16)
    assert bot_env.action_nvec.tolist() == [256, 6, 4, 4, 4, 4, 7, 49]
    trans = bot_env.reset()
    assert tuple(trans["obs"].shape) == (4, 27, 16, 16)
    assert trans["done"].dtype == torch.bool


def test_random_policy_action_codec_roundtrips(bot_env):
    policy = RandomPolicy(bot_env.action_nvec)
    trans = bot_env.reset()
    out = policy.step(trans["obs"])
    assert tuple(out["action"].shape) == (4, 8)
    # Actions stay within the MultiDiscrete bounds and step without error.
    assert (out["action"] < bot_env.action_nvec).all()
    nxt = bot_env.step(out["action"])
    assert tuple(nxt["obs"].shape) == (4, 27, 16, 16)


def test_collector_fills_buffer(bot_env):
    policy = RandomPolicy(bot_env.action_nvec)
    collector = Collector(bot_env, policy, horizon=8)
    buf = collector.collect()
    assert tuple(buf.data["obs"].shape) == (8, 4, 27, 16, 16)
    assert tuple(buf.data["action"].shape) == (8, 4, 8)
    assert torch.isfinite(buf.data["advantage"]).all()
    assert torch.isfinite(buf.data["return"]).all()


def test_selfplay_collection(selfplay_env):
    # Self-play exposes both player slots as ordinary envs; the same policy nets
    # all of them, so collection is identical to bot mode.
    policy = RandomPolicy(selfplay_env.action_nvec)
    collector = Collector(selfplay_env, policy, horizon=6)
    buf = collector.collect()
    assert tuple(buf.data["obs"].shape) == (6, 4, 27, 16, 16)
