"""``DreamEnv`` — collation wrapper adding Dreamer's ``is_first`` episode-start flag.

The sequence world model needs to know where episode boundaries fall so it can
reset its temporal context and so the continue head has a target. gym_microrts
auto-resets internally, so the wrapped ``MicroRTSVecEnv`` never surfaces the reset
explicitly — the observation after a ``done`` step is already the *next* episode's
first frame.

This wrapper tags every transition with ``is_first`` (True for the very first
post-reset observation, and thereafter equal to the previous step's ``done``, since
that is exactly when the returned obs is a fresh episode start). Everything else is
delegated to the wrapped env, so it drops into ``Collector``/``DreamCollector`` and
the existing eval path unchanged.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


class DreamEnv:
    def __init__(self, env) -> None:
        self.env = env
        self._pending_first: str | None = None  # "reset" | "step" | None

    # --- delegated attributes -------------------------------------------
    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def obs_shape(self):
        return self.env.obs_shape

    @property
    def action_nvec(self):
        return self.env.action_nvec

    @property
    def mask_shape(self):
        return getattr(self.env, "mask_shape", None)

    def __getattr__(self, name):
        # Fall through to the wrapped env for anything not overridden (e.g. gridnet,
        # cfg, close). Guarded so the wrapped handle itself resolves normally.
        return getattr(self.__dict__["env"], name)

    # --- async API with is_first tagging --------------------------------
    def async_reset(self, seed: int | None = None) -> None:
        self._pending_first = "reset"
        self.env.async_reset(seed)

    def send(self, actions: torch.Tensor) -> None:
        self._pending_first = "step"
        self.env.send(actions)

    def recv(self) -> TensorDict:
        td = self.env.recv()
        n = td.batch_size[0]
        if self._pending_first == "reset":
            is_first = torch.ones(n, dtype=torch.bool)
        else:
            # Post-step obs is an episode start exactly where the step ended one.
            is_first = td["done"].clone().bool()
        td.set("is_first", is_first)
        self._pending_first = None
        return td

    def reset(self, seed: int | None = None) -> TensorDict:
        self.async_reset(seed)
        return self.recv()

    def step(self, actions: torch.Tensor) -> TensorDict:
        self.send(actions)
        return self.recv()

    def close(self) -> None:
        self.env.close()
