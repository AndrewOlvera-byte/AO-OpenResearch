"""Phase 3: CNN policy, PPO update, and the overlap-toggleable Runner."""

import torch

from collectors.buffer import RolloutBuffer
from collectors.runner import RunConfig, Runner
from models.cnn_policy import CNNPolicy
from trainers.PPOTrainer import PPOTrainer

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def test_cnn_policy_step_and_evaluate():
    policy = CNNPolicy(OBS, NVEC, device="cpu")
    obs = torch.randn(4, *OBS)
    out = policy.step(obs)
    assert tuple(out["action"].shape) == (4, 8)
    assert (out["action"] < NVEC).all() and (out["action"] >= 0).all()

    logp, ent, value = policy.evaluate_actions(obs, out["action"])
    assert logp.shape == (4,) and ent.shape == (4,) and value.shape == (4,)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()


def test_ppo_update_runs_and_changes_weights():
    policy = CNNPolicy(OBS, NVEC, device="cpu")
    before = policy.actor.weight.detach().clone()
    buf = RolloutBuffer(horizon=4, num_envs=4, obs_shape=OBS, action_dim=8)
    buf.data["obs"] = torch.randn(4, 4, *OBS)
    buf.data["action"] = (torch.rand(4, 4, 8) * NVEC).long()
    buf.data["advantage"] = torch.randn(4, 4)
    buf.data["return"] = torch.randn(4, 4)
    PPOTrainer(policy, epochs=2, minibatches=2).update(buf)
    assert not torch.equal(before, policy.actor.weight)  # learner moved


def _cfg(overlap):
    return RunConfig(num_envs=4, horizon=4, iters=2, device="cpu",
                     policy="cnn", backend="serial", overlap=overlap,
                     epochs=1, minibatches=2)


def test_runner_sync():
    runner = Runner(_cfg(overlap=False))
    try:
        m = runner.run()
        assert m["sps"] > 0 and m["steps"] == 2 * 4 * 4
    finally:
        runner.close()


def test_runner_overlap():
    runner = Runner(_cfg(overlap=True))
    try:
        m = runner.run()
        assert m["sps"] > 0 and m["steps"] == 2 * 4 * 4
        assert runner.actor is not runner.learner  # snapshot used for collection
    finally:
        runner.close()
