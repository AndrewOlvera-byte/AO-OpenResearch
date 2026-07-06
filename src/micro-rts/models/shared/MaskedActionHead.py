"""``MaskedMultiDiscreteActionHead`` — invalid-action-masked head for MicroRTS.

MicroRTS (gym_microrts 0.3.2, ``MicroRTSGridModeVecEnv``) uses a *single-unit*
action per env step: ``nvec = [H*W, 6, 4, 4, 4, 4, 7, 49]``. Component 0 selects
which grid cell's unit to command; components 1..7 are that unit's action
(type + parameters). The overwhelming majority of the ``H*W`` source choices are
illegal (empty cells) and most component choices are illegal for a given unit, so
an *unmasked* policy samples almost entirely no-ops — entropy pins near its
theoretical max and nothing is learned. Invalid action masking is the single most
important ingredient for learning on MicroRTS (Huang & Ontanon 2020/2021).

The engine mask (surfaced by ``MicroRTSVecEnv`` from the JVM client's
``getMasks``) has shape ``(N, H*W, 1 + sum(component_nvec))`` = ``(N, H*W, 79)``:
channel 0 is the per-cell source-selectable flag; channels ``1:`` are the per-cell
component masks. This head:

1. produces **spatial** source logits (a 1x1 conv over the ``H*W`` feature map, so
   the source choice is position-aware), masks them with channel 0, and samples a
   source cell ``s``;
2. gathers the feature vector *at cell ``s``* -> component logits, masks each
   component with that cell's mask row, and samples the 7 components.

``sample`` and ``log_prob_entropy`` mask with the *same* stored mask, so the PPO
importance ratio is computed over identical supports (a hard requirement for a
correct update).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

NEG_INF = -1e8


class _MaskedCategorical:
    """Categorical with invalid actions masked out (à la CleanRL ``CategoricalMasked``).

    A row with **no** valid action (e.g. an irrelevant component like attack-target
    for a non-attacking unit, or a cell with no actable unit) is a *degenerate*
    no-choice: its logits are neutralized so the softmax stays finite, and it
    contributes **0** to both log-prob and entropy. That is the correct treatment —
    the engine ignores such components, so they must not add noise to the policy
    gradient or inflate the entropy bonus (opening them to uniform, the naive guard,
    dumps ~13 nats of meaningless entropy and trains junk logits)."""

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        valid = mask > 0.5
        self.any_valid = valid.any(dim=-1).float()          # (N,) 1 if a real choice
        logits = logits.masked_fill(~valid, NEG_INF)
        # Neutralize all-invalid rows so Categorical is finite (contribution zeroed below).
        logits = torch.where(valid.any(dim=-1, keepdim=True), logits, torch.zeros_like(logits))
        self.dist = Categorical(logits=logits)

    def sample(self, deterministic: bool = False) -> torch.Tensor:
        return self.dist.logits.argmax(-1) if deterministic else self.dist.sample()

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(action) * self.any_valid

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy() * self.any_valid


class MaskedMultiDiscreteActionHead(nn.Module):
    def __init__(self, feat_channels: int, nvec: torch.Tensor, logits_gain: float = 0.01) -> None:
        super().__init__()
        splits: list[int] = nvec.tolist()
        self.source_dim = int(splits[0])            # H*W
        self.comp_splits: list[int] = splits[1:]    # [6, 4, 4, 4, 4, 7, 49]
        # Spatial source head: one logit per cell, keeping position information.
        self.source_head = nn.Conv2d(feat_channels, 1, kernel_size=1)
        # Component head reads the chosen unit's own feature vector.
        self.comp_head = nn.Linear(feat_channels, sum(self.comp_splits))
        for layer in (self.source_head, self.comp_head):
            nn.init.orthogonal_(layer.weight, gain=logits_gain)
            nn.init.zeros_(layer.bias)

    def _source_dist(self, feat_map: torch.Tensor, mask: torch.Tensor) -> _MaskedCategorical:
        n = feat_map.shape[0]
        logits = self.source_head(feat_map).reshape(n, self.source_dim)  # row-major H*W
        return _MaskedCategorical(logits, mask[..., 0])

    def _component_dists(self, feat_map: torch.Tensor, source: torch.Tensor,
                         mask: torch.Tensor) -> list[_MaskedCategorical]:
        n, c = feat_map.shape[0], feat_map.shape[1]
        idx = torch.arange(n, device=feat_map.device)
        cell_feat = feat_map.reshape(n, c, -1)[idx, :, source]   # (N, C) at chosen cell
        comp_logits = self.comp_head(cell_feat)                  # (N, sum(comp_splits))
        comp_mask = mask[idx, source, 1:]                        # (N, sum(comp_splits))
        dists, off = [], 0
        for sz in self.comp_splits:
            dists.append(_MaskedCategorical(comp_logits[:, off:off + sz], comp_mask[:, off:off + sz]))
            off += sz
        return dists

    def sample(self, feat_map: torch.Tensor, mask: torch.Tensor,
               deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action[N, 1+len(comp_splits)], logprob[N])."""
        src_dist = self._source_dist(feat_map, mask)
        source = src_dist.sample(deterministic)
        logprob = src_dist.log_prob(source)

        actions = [source]
        for dist in self._component_dists(feat_map, source, mask):
            a = dist.sample(deterministic)
            logprob = logprob + dist.log_prob(a)
            actions.append(a)
        return torch.stack(actions, -1), logprob

    def log_prob_entropy(self, feat_map: torch.Tensor, action: torch.Tensor,
                         mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate stored (source + component) actions under the stored mask."""
        source = action[..., 0]
        src_dist = self._source_dist(feat_map, mask)
        logprob = src_dist.log_prob(source)
        entropy = src_dist.entropy()
        for i, dist in enumerate(self._component_dists(feat_map, source, mask)):
            logprob = logprob + dist.log_prob(action[..., 1 + i])
            entropy = entropy + dist.entropy()
        return logprob, entropy
