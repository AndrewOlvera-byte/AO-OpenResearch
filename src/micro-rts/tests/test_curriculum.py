"""PPO curriculum: phase schedule, opponent pool, and the in-process env rebuild."""

import torch

from environments.curriculum.OpponentPool import OpponentPool
from environments.curriculum.PPOCurriculum import PPOCurriculum
from models.cnn_mlp_policy import CNNMLPPolicy

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def test_phase_schedule_and_freeze():
    cur = PPOCurriculum(bot_steps=1000, freeze_steps=500, num_envs=4)
    assert cur.phase(0) == "bot"
    assert cur.phase(999) == "bot"
    assert cur.phase(1000) == "selfplay"
    assert cur.env_config(0).mode == "bot"
    assert cur.env_config(2000).mode == "selfplay"
    assert cur.should_rebuild_env(999, 1000)
    assert not cur.should_rebuild_env(0, 999)


def test_on_step_freeze_and_pool_growth():
    cur = PPOCurriculum(bot_steps=0, freeze_steps=100, snapshot_every=50, num_envs=4)
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    e0 = cur.on_step(0, policy)
    assert e0["phase"] == "selfplay" and e0["encoder_frozen"] is True
    assert e0["snapshot_pushed"] and e0["pool_size"] == 1
    # next step within snapshot_every -> no new push.
    e1 = cur.on_step(10, policy)
    assert not e1["snapshot_pushed"] and e1["encoder_frozen"] is True
    # past freeze_steps -> encoder unfrozen.
    e2 = cur.on_step(200, policy)
    assert e2["encoder_frozen"] is False
    assert all(p.requires_grad for p in policy.encoder.parameters())


def test_opponent_pool_fifo_and_sampling():
    pool = OpponentPool(capacity=2)
    assert pool.sample() is None
    pool.push({"w": torch.tensor([1.0])})
    pool.push({"w": torch.tensor([2.0])})
    pool.push({"w": torch.tensor([3.0])})  # evicts the first
    assert len(pool) == 2
    vals = {float(pool.sample()["w"]) for _ in range(50)}
    assert vals.issubset({2.0, 3.0}) and 3.0 in vals


def test_bot_to_selfplay_env_rebuild_in_one_process():
    """Guards the JVM multi-build risk: build bot env, rebuild to self-play, collect."""
    from collectors.collector import Collector
    from collectors.selfplay_collector import SelfPlayCollector
    from environments.microrts_env import MicroRTSVecEnv

    cur = PPOCurriculum(bot_steps=10, num_envs=4)
    learner = CNNMLPPolicy(OBS, NVEC, device="cpu")

    env = MicroRTSVecEnv(cur.env_config(0))
    buf = Collector(env, learner, horizon=3).collect()
    assert buf.data["obs"].shape[0] == 3

    # Rebuild to self-play WITHOUT closing the old env: gym_microrts shares one
    # JVM per process and JPype cannot restart it, so close() would kill the JVM.
    # The trainer relies on this same "build new, drop old" behavior at the phase
    # boundary.
    env2 = MicroRTSVecEnv(cur.env_config(100))  # self-play
    opp = CNNMLPPolicy(OBS, NVEC, device="cpu")
    buf = SelfPlayCollector(env2, learner, opp, horizon=3).collect()
    assert buf.data["obs"].shape[1] == env2.num_envs // 2  # learner lanes only
    env2.close()
