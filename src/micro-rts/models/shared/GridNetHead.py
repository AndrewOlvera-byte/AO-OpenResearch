"""``GridNetActionHead`` — per-cell masked action head (canonical Gym-uRTS GridNet).

Unlike the single-unit head (pick one cell, then its action), GridNet emits an action
for **every** grid cell each step: for each of the ``H*W`` cells a 7-component
unit-action ``[6,4,4,4,4,7,49]``, masked by that cell's engine mask. Cells with no
actable unit have an all-zero mask, so ``_MaskedCategorical`` collapses them to a NOOP
that contributes nothing to the log-prob, entropy, or gradient — exactly what we want
(they are ignored by the engine). The per-env log-prob / entropy is the sum over cells.

Reuses ``_MaskedCategorical`` from :mod:`MaskedActionHead` so the masking / degenerate
handling is identical to the single-unit head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .MaskedActionHead import _MaskedCategorical


class GridNetActionHead(nn.Module):
    def __init__(self, cell_nvec: torch.Tensor) -> None:
        super().__init__()
        self.comp_splits: list[int] = cell_nvec.tolist()   # [6, 4, 4, 4, 4, 7, 49]
        self.n_comp = len(self.comp_splits)

    def _dists(self, cell_logits: torch.Tensor, comp_mask: torch.Tensor) -> list[_MaskedCategorical]:
        dists, off = [], 0
        for sz in self.comp_splits:
            dists.append(_MaskedCategorical(cell_logits[:, off:off + sz], comp_mask[:, off:off + sz]))
            off += sz
        return dists

    def sample(self, cell_logits: torch.Tensor, mask: torch.Tensor,
               deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """cell_logits (N, H*W, sum(comp)); mask (N, H*W, 1+sum(comp)). Returns
        (action[N, H*W, 7], logprob[N])."""
        n, hw, _ = cell_logits.shape
        flat = cell_logits.reshape(n * hw, -1)
        comp_mask = mask[..., 1:].reshape(n * hw, -1)      # drop the source-flag channel
        logp = flat.new_zeros(n * hw)
        actions = []
        for dist in self._dists(flat, comp_mask):
            a = dist.sample(deterministic)
            logp = logp + dist.log_prob(a)
            actions.append(a)
        action = torch.stack(actions, -1).reshape(n, hw, self.n_comp)
        return action, logp.reshape(n, hw).sum(-1)

    def log_prob_entropy(self, cell_logits: torch.Tensor, action: torch.Tensor,
                         mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n, hw, _ = cell_logits.shape
        flat = cell_logits.reshape(n * hw, -1)
        comp_mask = mask[..., 1:].reshape(n * hw, -1)
        act = action.reshape(n * hw, self.n_comp)
        logp = flat.new_zeros(n * hw)
        ent = flat.new_zeros(n * hw)
        for i, dist in enumerate(self._dists(flat, comp_mask)):
            logp = logp + dist.log_prob(act[:, i])
            ent = ent + dist.entropy()
        return logp.reshape(n, hw).sum(-1), ent.reshape(n, hw).sum(-1)
