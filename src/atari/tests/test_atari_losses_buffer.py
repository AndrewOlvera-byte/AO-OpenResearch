"""Atari loss + buffer/collector tests (no ALE for losses; small ALE for collector)."""

import torch

import models.dreamer as D
from loss.dreamer import (
    PercentileReturnNormalizer,
    world_model_loss, shortcut_forcing_loss, lambda_return, actor_critic_losses,
)
from collectors.sequence_buffer import SequenceReplayBuffer
from collectors.dream_collector import DreamCollector
from environments.atari_env import AtariVecEnv, AtariEnvConfig

OBS = (1, 64, 64)
NA = 6


def _model():
    cfg = D.AtariDreamerConfig.from_dict({
        "tokenizer": {"base_channels": 16, "d_latent": 16},
        "dynamics": {"d_model": 32, "depth": 2, "n_heads": 4, "n_register": 2},
        "actor_critic": {"hidden": [64], "imagine_horizon": 4},
    })
    return D.AtariDreamerV4(OBS, NA, cfg)


def _batch(B=2, T=4):
    return {
        "obs": torch.rand(B, T, *OBS),
        "action": torch.randint(0, NA, (B, T)),
        "reward": torch.randn(B, T),
        "cont": torch.ones(B, T),
        "is_first": torch.zeros(B, T, dtype=torch.bool),
    }


def test_lambda_return_hand_computed():
    reward = torch.tensor([[0.0, 0.0]])
    value = torch.tensor([[1.0, 2.0, 3.0]])
    disc = torch.tensor([[1.0, 1.0]])
    out = lambda_return(reward, value, disc, lam=0.5)
    assert torch.allclose(out, torch.tensor([[2.5, 3.0]]), atol=1e-6)


def test_percentile_return_normalizer_has_sparse_reward_floor():
    norm = PercentileReturnNormalizer(limit=1.0)
    scale, metrics = norm.scale(torch.zeros(8, 4))
    assert torch.allclose(scale, torch.tensor(1.0))
    assert metrics["ac/return_norm_scale"] == 1.0
    scale, metrics = norm.scale(torch.linspace(-4, 4, 32).reshape(8, 4))
    assert float(scale) >= 1.0
    assert metrics["ac/return_norm_high"] > metrics["ac/return_norm_low"]


def test_world_loss_trains_only_world():
    m = _model()
    b = _batch()
    loss, metrics, z = world_model_loss(
        m, b["obs"], b["action"], b["reward"], b["cont"], b["is_first"]
    )
    assert torch.isfinite(loss)
    assert {"wm/recon", "wm/dynamics", "wm/reward", "wm/continue"} <= set(metrics)
    loss.backward()
    assert m.tokenizer.to_latent.weight.grad is not None
    assert m.world_model.next_head.weight.grad is not None
    assert m.action_expert.actor.net[0].weight.grad is None
    assert m.action_expert.critic.net[0].weight.grad is None


def test_world_loss_masks_reset_boundaries():
    m = _model()
    b = _batch(B=1, T=4)
    b["is_first"][0, 2] = True
    loss, metrics, _ = world_model_loss(
        m, b["obs"], b["action"], b["reward"], b["cont"], b["is_first"]
    )
    assert torch.isfinite(loss)
    assert abs(metrics["wm/dynamics_valid_frac"] - 2 / 3) < 1e-6


def test_shortcut_forcing_finite_and_encoder_untouched():
    m = _model()
    b = _batch()
    z = m.tokenizer.encode(b["obs"]).detach()
    loss, metrics = shortcut_forcing_loss(m, z, b["action"], b["is_first"])
    assert torch.isfinite(loss) and metrics["flow/matching"] >= 0
    loss.backward()
    assert m.world_model.flow_head[0].weight.grad is not None
    assert m.tokenizer.to_latent.weight.grad is None


def test_actor_critic_routes_grads_and_spares_world():
    m = _model()
    z0 = torch.rand(6, m.tokenizer.n_spatial, m.tokenizer.d_latent)
    imagined = m.imagine(z0, horizon=4)
    norm = PercentileReturnNormalizer(limit=1.0)
    actor_loss, critic_loss, metrics = actor_critic_losses(
        m, imagined, normalize_adv=False, return_normalizer=norm
    )
    assert torch.isfinite(actor_loss) and torch.isfinite(critic_loss)
    assert metrics["ac/return_norm_scale"] >= 1.0
    actor_loss.backward(retain_graph=True)
    critic_loss.backward()
    assert m.action_expert.actor.net[0].weight.grad is not None
    assert m.action_expert.critic.net[0].weight.grad is not None
    assert m.world_model.next_head.weight.grad is None
    assert all(p.grad is None for p in m.action_expert.target_critic.parameters())


def test_sequence_buffer_contiguous_and_ring():
    buf = SequenceReplayBuffer(capacity=6, num_envs=2, obs_shape=OBS)
    for t in range(8):
        buf.add(obs=torch.full((2, *OBS), float(t)), action=torch.zeros(2, dtype=torch.long),
                reward=torch.ones(2), cont=torch.ones(2), is_first=torch.zeros(2, dtype=torch.bool))
    assert buf.size == 6
    assert buf.num_transitions == 12
    batch = buf.sample(batch=5, seq_len=3)
    assert batch["obs"].shape == (5, 3, *OBS)
    vals = batch["obs"].reshape(5, 3, -1)[..., 0]
    assert torch.all(vals[:, 1:] - vals[:, :-1] == 1)   # consecutive timesteps
    assert vals.min() >= 2                              # oldest overwritten


def test_sequence_buffer_avoids_reset_boundaries():
    buf = SequenceReplayBuffer(capacity=8, num_envs=1, obs_shape=OBS)
    for t in range(8):
        buf.add(obs=torch.full((1, *OBS), float(t)), action=torch.zeros(1, dtype=torch.long),
                reward=torch.ones(1), cont=torch.ones(1),
                is_first=torch.tensor([t == 4]))
    batch = buf.sample(batch=8, seq_len=3)
    assert not bool(batch["is_first"][:, 1:].any())


def test_dream_collector_fills_buffer():
    env = AtariVecEnv(AtariEnvConfig(game="pong", num_envs=3, max_steps=50))
    try:
        coll = DreamCollector(env, _model(), horizon=5, capacity=32)
        buf = coll.collect()
        assert buf.size == 5
        coll.collect()
        assert buf.size == 10
        batch = buf.sample(batch=4, seq_len=4)
        assert batch["action"].shape == (4, 4) and batch["obs"].shape == (4, 4, *OBS)
    finally:
        env.close()
