from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IntentPriorConfig


class MultimodalOpponentIntentPrior(nn.Module):
    """History-conditioned finite mixture over privileged opponent plans.

    Each mode is a complete eight-token plan hypothesis.  Keeping modes
    explicit makes best-of-N coverage, calibration, and policy-facing entropy
    directly measurable, unlike stochastic variation hidden inside a large
    joint state/intent flow.
    """

    def __init__(
        self,
        history_dim: int = 512,
        n_plan_tokens: int = 8,
        plan_dim: int = 320,
        cfg: IntentPriorConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or IntentPriorConfig()
        self.n_modes = int(self.cfg.n_modes)
        self.n_plan_tokens = int(n_plan_tokens)
        self.plan_dim = int(plan_dim)
        d = int(self.cfg.d_model)
        self.history_in = nn.Linear(history_dim, d)
        self.mode_embed = nn.Parameter(torch.zeros(self.n_modes, d))
        self.plan_position = nn.Parameter(torch.zeros(self.n_plan_tokens, d))
        self.mode_plan_bias = nn.Parameter(
            torch.empty(self.n_modes, self.n_plan_tokens, self.plan_dim)
        )
        nn.init.normal_(self.mode_embed, std=0.02)
        nn.init.normal_(self.plan_position, std=0.02)
        nn.init.normal_(self.mode_plan_bias, std=0.5)
        layer = nn.TransformerDecoderLayer(
            d,
            self.cfg.n_heads,
            int(d * self.cfg.mlp_ratio),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, self.cfg.depth, nn.LayerNorm(d))
        self.plan_out = nn.Linear(d, self.plan_dim)
        self.mode_head = nn.Sequential(nn.LayerNorm(history_dim), nn.Linear(history_dim, self.n_modes))
        self.history_contrast = nn.Sequential(
            nn.LayerNorm(history_dim), nn.Linear(history_dim, self.cfg.contrast_dim)
        )
        self.plan_contrast = nn.Sequential(
            nn.LayerNorm(self.plan_dim), nn.Linear(self.plan_dim, self.cfg.contrast_dim)
        )

    def forward(self, history_registers):
        lead = history_registers.shape[:-2]
        memory_raw = history_registers.reshape(-1, *history_registers.shape[-2:])
        memory = self.history_in(memory_raw)
        queries = (
            self.mode_embed[:, None, :] + self.plan_position[None, :, :]
        ).reshape(1, self.n_modes * self.n_plan_tokens, -1)
        queries = queries.expand(memory.shape[0], -1, -1)
        hidden = self.decoder(queries, memory)
        plans = self.plan_out(hidden).reshape(
            *lead, self.n_modes, self.n_plan_tokens, self.plan_dim
        ) + self.mode_plan_bias
        pooled = memory_raw.mean(-2)
        logits = self.mode_head(pooled).reshape(*lead, self.n_modes)
        history_embedding = F.normalize(
            self.history_contrast(pooled), dim=-1
        ).reshape(*lead, -1)
        return plans, logits, history_embedding

    def target_embedding(self, normalized_plan):
        return F.normalize(self.plan_contrast(normalized_plan.mean(-2)), dim=-1)

    def select(self, plans, mode_logits, *, sample=False):
        if sample:
            index = torch.distributions.Categorical(logits=mode_logits).sample()
        else:
            index = mode_logits.argmax(-1)
        gather = index[..., None, None, None].expand(
            *index.shape, 1, self.n_plan_tokens, self.plan_dim
        )
        return plans.gather(-3, gather).squeeze(-3), index
