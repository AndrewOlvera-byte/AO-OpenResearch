"""Reward normalization wrapper + running mean/std."""

import torch
from tensordict import TensorDict

from environments.normalize import NormalizeReward, RunningMeanStd


def test_running_mean_std_matches_torch():
    rms = RunningMeanStd()
    data = torch.randn(1000)
    for chunk in data.chunk(10):
        rms.update(chunk)
    assert abs(rms.mean - float(data.mean())) < 1e-2
    assert abs(rms.var - float(data.var(unbiased=False))) < 1e-2


class _StubEnv:
    """Minimal VecEnv-shaped stub emitting a fixed reward each step."""

    def __init__(self, n=4, reward=5.0):
        self.num_envs = n
        self.obs_shape = (3, 4, 4)
        self.action_nvec = torch.tensor([2, 2])
        self._r = reward

    def async_reset(self, seed=None):
        self._pending = self._pack(0.0, done=False)

    def send(self, actions):
        self._pending = self._pack(self._r, done=False)

    def recv(self):
        return self._pending

    def _pack(self, r, done):
        n = self.num_envs
        return TensorDict(
            {
                "obs": torch.zeros(n, *self.obs_shape),
                "reward": torch.full((n,), float(r)),
                "done": torch.full((n,), done, dtype=torch.bool),
            },
            batch_size=[n],
        )

    def close(self):
        pass


def test_normalize_reward_scales_and_passes_through():
    env = NormalizeReward(_StubEnv(reward=5.0), gamma=0.99, clip=10.0)
    env.reset()  # reset frame returns raw (dummy) reward, not normalized
    scales = []
    for _ in range(50):
        trans = env.step(torch.zeros(4, 2))
        scales.append(float(trans["reward"].abs().mean()))
    # Constant raw reward -> after dividing by the growing return-std, the
    # normalized magnitude is finite, clipped, and not the raw 5.0.
    assert all(s <= 10.0 for s in scales)
    assert scales[-1] != 5.0
    # obs/done are passed through untouched.
    assert trans["obs"].shape == (4, 3, 4, 4)


def test_normalize_reward_resets_return_on_done():
    env = NormalizeReward(_StubEnv(reward=1.0), gamma=0.99)
    env.reset()
    env.step(torch.zeros(4, 2))
    before = env._returns.clone()
    assert (before > 0).all()
    env.env._pending = env.env._pack(1.0, done=True)  # force a terminal
    env._is_reset = False
    env.recv()
    assert (env._returns == 0).all()  # accumulator cleared at episode boundary
