"""``WorldModelMemory`` — rolling sequence/history context for the world model.

The transformer world model has no recurrent state: its memory *is* the window of
past ``(latent, action, is_first)`` frames it attends over. During real-env
collection the policy acts frame-by-frame, so something has to accumulate that
window — this wrapper does, per vectorized env batch:

- ``append(z, action, is_first)`` after every env step (the collector calls it
  with the ``z`` the policy already computed in ``step``),
- a bounded window: only the last ``context_len`` frames are kept,
- episode boundaries are **not** spliced out — ``is_first`` flags stay in the
  window and the world model consumes them natively (it replaces the action
  signal at episode starts), exactly like sampled replay sequences.

``context()`` returns the stacked ``(N, T, ...)`` tensors ready to feed
``WorldModel.contextualize`` / ``DreamerV4.imagine(...)`` — e.g. to seed
imagination from *live* on-policy state or to run dynamics analysis on fresh
trajectories.
"""

from __future__ import annotations

import torch


class WorldModelMemory:
    def __init__(self, context_len: int) -> None:
        self.context_len = int(context_len)
        self._z: list[torch.Tensor] = []
        self._action: list[torch.Tensor] = []
        self._is_first: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._z)

    @property
    def ready(self) -> bool:
        return len(self._z) > 0

    def reset(self) -> None:
        self._z.clear()
        self._action.clear()
        self._is_first.clear()

    def append(self, z: torch.Tensor, action: torch.Tensor,
               is_first: torch.Tensor) -> None:
        """z (N,n_spatial,d_latent), action (N,H*W,7), is_first (N,) — one env step."""
        self._z.append(z.detach())
        self._action.append(action.detach())
        self._is_first.append(is_first.detach().bool())
        if len(self._z) > self.context_len:
            self._z.pop(0)
            self._action.pop(0)
            self._is_first.pop(0)

    def context(self) -> dict[str, torch.Tensor]:
        """Stacked window: ``z`` (N,T,...), ``action`` (N,T,...), ``is_first`` (N,T)."""
        assert self.ready, "empty world-model memory"
        return {
            "z": torch.stack(self._z, dim=1),
            "action": torch.stack(self._action, dim=1),
            "is_first": torch.stack(self._is_first, dim=1),
        }
