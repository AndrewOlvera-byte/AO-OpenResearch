"""Reward component surfacing, ShapeReward weighting, and curriculum annealing."""

import torch
from tensordict import TensorDict

from environments.curriculum.PPOCurriculum import PPOCurriculum
from environments.shape_reward import ShapeReward
from rewards.rewards import lerp_weights, reward_weight


class _RawEnv:
    """Stub emitting a fixed 6-component raw_rewards vector per env."""

    num_envs = 2
    obs_shape = (3, 4, 4)
    action_nvec = torch.tensor([2, 2])

    def __init__(self, raw):
        self._raw = torch.tensor(raw, dtype=torch.float32)

    def _pack(self):
        return TensorDict(
            {
                "obs": torch.zeros(2, 3, 4, 4),
                "reward": torch.zeros(2),
                "raw_rewards": self._raw,
                "done": torch.zeros(2, dtype=torch.bool),
            },
            batch_size=[2],
        )

    def async_reset(self, seed=None):
        self._p = self._pack()

    def send(self, a):
        self._p = self._pack()

    def recv(self):
        return self._p

    def close(self):
        pass


def test_lerp_weights():
    a = (0.0, 10.0)
    b = (10.0, 0.0)
    assert lerp_weights(a, b, 0.0) == (0.0, 10.0)
    assert lerp_weights(a, b, 1.0) == (10.0, 0.0)
    assert lerp_weights(a, b, 0.5) == (5.0, 5.0)
    assert lerp_weights(a, b, 2.0) == (10.0, 0.0)  # clamped


def test_shape_reward_applies_and_updates_weight():
    # raw: [win=+1 resource=5 ...], [win=-1 resource=3 ...]
    env = ShapeReward(_RawEnv([[1, 5, 0, 0, 0, 0], [-1, 3, 0, 0, 0, 0]]),
                      weight=reward_weight("win_loss"))
    trans = env.step(torch.zeros(2, 2))
    assert torch.allclose(trans["reward"], torch.tensor([1.0, -1.0]))  # win-loss only

    env.set_weight(reward_weight("dense_shaped"))  # win=10, resource=1
    trans = env.step(torch.zeros(2, 2))
    assert torch.allclose(trans["reward"], torch.tensor([10.0 + 5.0, -10.0 + 3.0]))


def test_curriculum_reward_annealing():
    cur = PPOCurriculum(reward_start_preset="dense_shaped",
                        reward_end_preset="win_focused",
                        reward_anneal_steps=1000, num_envs=4)
    def close(a, b):
        return all(abs(x - y) < 1e-9 for x, y in zip(a, b))

    assert close(cur.reward_weight_vec(0), reward_weight("dense_shaped"))
    assert close(cur.reward_weight_vec(1000), reward_weight("win_focused"))
    # resource component (index 1) decays 1.0 -> 0.05 halfway.
    mid = cur.reward_weight_vec(500)
    assert abs(mid[1] - (1.0 + (0.05 - 1.0) * 0.5)) < 1e-6
