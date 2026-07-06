"""Core RL data-plane interfaces.

Two abstractions the whole stack depends on:

- ``VecEnv``  — an async-capable vectorized env (PufferLib-style ``async_reset`` /
  ``send`` / ``recv``). Concrete impls: ``MicroRTSVecEnv`` (in-process, serial) and,
  later, a multiprocess shared-memory pool — both behind this one interface.
- ``Policy``  — anything that maps a batch of observations to actions. Both the
  ``RandomPolicy`` stub and a future ``nn.Module`` satisfy it structurally.

All payloads are ``tensordict.TensorDict`` so batched slicing / ``.to(device)`` /
minibatching come for free.

Conventions
-----------
Transition (returned by ``recv``): keys ``obs``[N,*obs_shape], ``reward``[N],
``done``[N] bool, ``trunc``[N] bool, ``env_id``[N]; optional ``mask``.
PolicyOutput (returned by ``Policy.step``): ``action``[N,A] long, ``logprob``[N],
``value``[N].
"""

from __future__ import annotations

import abc
from typing import Protocol, runtime_checkable

import torch
from tensordict import TensorDict


@runtime_checkable
class Policy(Protocol):
    def step(self, obs: torch.Tensor, mask: torch.Tensor | None = None) -> TensorDict:
        """Map obs[N,*obs_shape] -> {action[N,A], logprob[N], value[N]}."""
        ...


class VecEnv(abc.ABC):
    """Async vectorized env. ``reset``/``step`` are sync sugar over the async API."""

    num_envs: int
    obs_shape: tuple[int, ...]          # channels-first, e.g. (27, 16, 16)
    action_nvec: torch.Tensor           # MultiDiscrete sizes, shape [A]

    @abc.abstractmethod
    def async_reset(self, seed: int | None = None) -> None: ...

    @abc.abstractmethod
    def send(self, actions: torch.Tensor) -> None: ...

    @abc.abstractmethod
    def recv(self) -> TensorDict: ...

    def reset(self, seed: int | None = None) -> TensorDict:
        self.async_reset(seed)
        return self.recv()

    def step(self, actions: torch.Tensor) -> TensorDict:
        self.send(actions)
        return self.recv()

    def close(self) -> None:  # noqa: B027 - optional override
        pass
