"""Clipped PPO loss + diagnostics."""

import torch

from collectors.buffer import RolloutBuffer
from loss.ppo import ppo_loss
from models.cnn_mlp_policy import CNNMLPPolicy

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def _minibatch(policy):
    buf = RolloutBuffer(horizon=4, num_envs=4, obs_shape=OBS, action_dim=8)
    obs = torch.randn(4, 4, *OBS)
    buf.data["obs"] = obs
    out = policy.step(obs.reshape(16, *OBS))
    buf.data["action"] = out["action"].reshape(4, 4, 8)
    buf.data["logprob"] = out["logprob"].reshape(4, 4)
    buf.data["value"] = out["value"].reshape(4, 4)
    buf.data["advantage"] = torch.randn(4, 4)
    buf.data["return"] = torch.randn(4, 4)
    return next(buf.minibatches(1))


def test_ppo_loss_returns_finite_metrics():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    loss, m = ppo_loss(_minibatch(policy), policy)
    assert torch.isfinite(loss)
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac", "total_loss"):
        assert key in m and m[key] == m[key]


def test_zero_kl_when_policy_unchanged():
    # Evaluating with the *same* params that produced logprob -> ratio == 1.
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    mb = _minibatch(policy)
    _, m = ppo_loss(mb, policy)
    assert abs(m["approx_kl"]) < 1e-5
    assert m["clipfrac"] == 0.0


def test_value_clipping_path():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    mb = _minibatch(policy)
    _, m_clip = ppo_loss(mb, policy, clip_vloss=True)
    _, m_noclip = ppo_loss(mb, policy, clip_vloss=False)
    assert torch.isfinite(torch.tensor(m_clip["value_loss"]))
    assert torch.isfinite(torch.tensor(m_noclip["value_loss"]))
