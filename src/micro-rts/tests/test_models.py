"""ResNet encoder, action head, and the CNNMLPPolicy."""

import torch

from models.cnn_mlp_policy import CNNMLPPolicy
from models.shared.ActionHead import MultiDiscreteActionHead
from models.shared.Encoder import ResNetEncoder

NVEC = torch.tensor([256, 6, 4, 4, 4, 4, 7, 49])
OBS = (27, 16, 16)


def test_resnet_encoder_shape():
    enc = ResNetEncoder(in_channels=OBS[0])
    out = enc(torch.randn(4, *OBS))
    assert out.shape == (4, enc.out_dim)
    assert torch.isfinite(out).all()


def test_action_head_sample_and_eval():
    head = MultiDiscreteActionHead(64, NVEC)
    feat = torch.randn(5, 64)
    action, logp = head.sample(feat)
    assert action.shape == (5, 8) and logp.shape == (5,)
    assert (action < NVEC).all() and (action >= 0).all()
    logp2, ent = head.log_prob_entropy(feat, action)
    assert logp2.shape == (5,) and ent.shape == (5,)
    assert torch.allclose(logp, logp2, atol=1e-5)


def test_policy_step_and_evaluate():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    obs = torch.randn(4, *OBS)
    out = policy.step(obs)
    assert tuple(out["action"].shape) == (4, 8)
    assert (out["action"] < NVEC).all() and (out["action"] >= 0).all()

    logp, ent, value = policy.evaluate_actions(obs, out["action"])
    assert logp.shape == (4,) and ent.shape == (4,) and value.shape == (4,)
    assert torch.isfinite(logp).all() and torch.isfinite(value).all()


def test_freeze_encoder_blocks_grads():
    policy = CNNMLPPolicy(OBS, NVEC, device="cpu")
    policy.freeze_encoder(True)
    logp, ent, value = policy.evaluate_actions(torch.randn(4, *OBS), torch.zeros(4, 8, dtype=torch.long))
    (logp.sum() + value.sum()).backward()
    enc_grads = [p.grad for p in policy.encoder.parameters() if p.grad is not None]
    assert enc_grads == []  # no encoder grads while frozen
    assert any(p.grad is not None for p in policy.neck.parameters())

    policy.unfreeze_encoder()
    assert all(p.requires_grad for p in policy.encoder.parameters())
