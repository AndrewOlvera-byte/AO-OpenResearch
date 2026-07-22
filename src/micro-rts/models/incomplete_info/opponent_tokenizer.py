from __future__ import annotations

import torch
import torch.nn as nn

from models.dreamer_v2.action_tokenizer import FactorizedActionEventEncoder

from .config import OpponentPlanTokenizerConfig
from .events import opponent_action_events
from .heads import EventFieldDecoder


class OpponentPlanTokenizer(nn.Module):
    """Privileged future trajectory -> compact plan tokens and exact events."""

    def __init__(
        self,
        n_state_tokens: int = 195,
        grid_hw=(16, 16),
        cfg: OpponentPlanTokenizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or OpponentPlanTokenizerConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        self.n_state_tokens = int(n_state_tokens)
        d = self.cfg.d_model
        self.event_encoder = FactorizedActionEventEncoder(
            d, self.grid_hw, self.cfg.max_unit_types, self.cfg.field_dim
        )
        self.horizon_embed = nn.Embedding(len(self.cfg.horizons), d)
        self.event_position = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        self.plan_queries = nn.Parameter(torch.zeros(self.cfg.n_plan_tokens, d))
        nn.init.normal_(self.event_position, std=0.02)
        nn.init.normal_(self.plan_queries, std=0.02)
        self.plan_cross = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d,
            self.cfg.n_heads,
            int(d * self.cfg.mlp_ratio),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.plan_transformer = nn.TransformerEncoder(
            layer, self.cfg.depth, nn.LayerNorm(d)
        )
        self.to_latent = nn.Linear(d, self.cfg.d_latent)
        self.from_latent = nn.Linear(self.cfg.d_latent, d)

        slots = len(self.cfg.horizons) * self.cfg.max_action_events
        self.decode_queries = nn.Parameter(torch.zeros(slots, d))
        nn.init.normal_(self.decode_queries, std=0.02)
        self.decode_cross = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        self.event_decoder = EventFieldDecoder(d, self.grid_hw, self.cfg.max_unit_types)
        self.state_queries = nn.Parameter(torch.zeros(self.n_state_tokens, d))
        nn.init.normal_(self.state_queries, std=0.02)
        self.state_cross = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        self.state_head = nn.Linear(d, self.cfg.d_latent)

    @property
    def max_horizon(self):
        return max(self.cfg.horizons)

    def target_events(self, state, opponent_action):
        anchors = state.shape[1] - self.max_horizon
        if anchors <= 0:
            raise ValueError(
                f"sequence length {state.shape[1]} must exceed max horizon "
                f"{self.max_horizon}"
            )
        events, valids, overflows = [], [], []
        for horizon in self.cfg.horizons:
            use_state = state[:, horizon : horizon + anchors]
            use_action = opponent_action[:, horizon : horizon + anchors]
            ev, va, ov = opponent_action_events(
                use_state,
                use_action,
                grid_hw=self.grid_hw,
                max_events=self.cfg.max_action_events,
            )
            events.append(ev)
            valids.append(va)
            overflows.append(ov)
        return (
            torch.stack(events, dim=2),
            torch.stack(valids, dim=2),
            torch.stack(overflows, dim=2),
        )

    def encode(self, state, opponent_action):
        events, valid, overflow = self.target_events(state, opponent_action)
        b, anchors, nh, ne = events.shape[:4]
        flat_events = events.reshape(b * anchors, nh, ne, -1)
        encoded = []
        for index in range(nh):
            item = self.event_encoder(flat_events[:, index])
            item = item + self.event_position[:ne] + self.horizon_embed.weight[index]
            encoded.append(item)
        context = torch.cat(encoded, dim=1)
        context_valid = valid.reshape(b * anchors, nh * ne)
        queries = self.plan_queries[None].expand(b * anchors, -1, -1)
        plan = (
            queries
            + self.plan_cross(
                queries,
                context,
                context,
                key_padding_mask=~context_valid,
                need_weights=False,
            )[0]
        )
        plan = self.to_latent(self.plan_transformer(plan))
        return (
            plan.reshape(b, anchors, self.cfg.n_plan_tokens, self.cfg.d_latent),
            events,
            valid,
            overflow,
        )

    def decode_events(self, plan):
        lead = plan.shape[:-2]
        flat = self.from_latent(plan.reshape(-1, *plan.shape[-2:]))
        queries = self.decode_queries[None].expand(flat.shape[0], -1, -1)
        decoded_tokens = (
            queries + self.decode_cross(queries, flat, flat, need_weights=False)[0]
        )
        logits = self.event_decoder(decoded_tokens)
        nh, ne = len(self.cfg.horizons), self.cfg.max_action_events
        return {
            name: value.reshape(*lead, nh, ne, *value.shape[2:])
            for name, value in logits.items()
        }

    def predict_future_state(self, plan):
        lead = plan.shape[:-2]
        flat = self.from_latent(plan.reshape(-1, *plan.shape[-2:]))
        queries = self.state_queries[None].expand(flat.shape[0], -1, -1)
        state = queries + self.state_cross(queries, flat, flat, need_weights=False)[0]
        return self.state_head(state).reshape(
            *lead, self.n_state_tokens, self.cfg.d_latent
        )
