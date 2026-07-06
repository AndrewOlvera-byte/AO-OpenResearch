"""RolloutBuffer storage + GAE correctness (no env / JVM needed)."""

import torch

from collectors.buffer import RolloutBuffer


def test_buffer_add_and_shapes():
    buf = RolloutBuffer(horizon=3, num_envs=2, obs_shape=(27, 16, 16), action_dim=8)
    buf.add(
        0,
        obs=torch.ones(2, 27, 16, 16),
        action=torch.zeros(2, 8, dtype=torch.long),
        logprob=torch.full((2,), -1.0),
        value=torch.full((2,), 0.5),
        reward=torch.ones(2),
        done=torch.zeros(2, dtype=torch.bool),
    )
    assert buf.data["obs"][0].eq(1).all()
    assert buf.data["value"][0].eq(0.5).all()
    assert tuple(buf.data["action"].shape) == (3, 2, 8)


def test_gae_matches_hand_computation():
    buf = RolloutBuffer(horizon=3, num_envs=1, obs_shape=(1,), action_dim=1)
    buf.data["reward"][:, 0] = torch.tensor([1.0, 0.0, 2.0])
    buf.data["value"][:, 0] = torch.tensor([0.5, 0.0, 1.0])
    buf.data["done"][:] = False
    buf.compute_gae(last_value=torch.zeros(1), gamma=0.99, lam=0.95)

    adv = buf.data["advantage"][:, 0]
    expected = torch.tensor([2.31562, 1.9305, 1.0])  # hand-computed GAE
    assert torch.allclose(adv, expected, atol=1e-4), adv
    ret = buf.data["return"][:, 0]
    assert torch.allclose(ret, adv + buf.data["value"][:, 0])


def test_minibatches_cover_all_samples():
    buf = RolloutBuffer(horizon=4, num_envs=3, obs_shape=(2,), action_dim=1)
    seen = sum(mb.batch_size[0] for mb in buf.minibatches(num_minibatches=4))
    assert seen == 4 * 3
