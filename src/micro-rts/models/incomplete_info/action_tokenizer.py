from __future__ import annotations

import torch
import torch.nn as nn

from models.dreamer_v2.action_tokenizer import FactorizedActionEventEncoder

from .config import SelfActionTokenizerConfig
from .events import self_action_events
from .heads import EventFieldDecoder


class SelfActionTokenizer(nn.Module):
    def __init__(
        self,
        n_state_tokens: int = 195,
        grid_hw=(16, 16),
        cfg: SelfActionTokenizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or SelfActionTokenizerConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        self.n_state_tokens = int(n_state_tokens)
        d = self.cfg.d_model
        self.event_encoder = FactorizedActionEventEncoder(
            d, self.grid_hw, self.cfg.max_unit_types, self.cfg.field_dim
        )
        self.event_position = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        nn.init.normal_(self.event_position, std=0.02)
        self.to_latent = nn.Linear(d, self.cfg.d_latent)
        self.from_latent = nn.Linear(self.cfg.d_latent, d)
        self.decoder = EventFieldDecoder(d, self.grid_hw, self.cfg.max_unit_types)
        self.state_query = nn.Linear(self.cfg.d_latent, d)
        self.effect_attn = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        self.effect_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, int(d * self.cfg.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d * self.cfg.mlp_ratio), self.cfg.d_latent),
        )

    def events(self, local_obs, action):
        return self_action_events(
            local_obs, action, max_events=self.cfg.max_action_events
        )

    def encode_events(self, events):
        h = self.event_encoder(events) + self.event_position[: events.shape[-2]]
        return self.to_latent(h)

    def decode_events(self, tokens):
        return self.decoder(self.from_latent(tokens))

    def predict_effect(self, state_tokens, action_tokens, valid):
        lead = state_tokens.shape[:-2]
        state = state_tokens.reshape(-1, *state_tokens.shape[-2:])
        actions = action_tokens.reshape(-1, *action_tokens.shape[-2:])
        mask = valid.reshape(-1, valid.shape[-1])
        query = self.state_query(state)
        routed = self.effect_attn(
            query,
            self.from_latent(actions),
            self.from_latent(actions),
            key_padding_mask=~mask,
            need_weights=False,
        )[0]
        return self.effect_head(query + routed).reshape(
            *lead, self.n_state_tokens, self.cfg.d_latent
        )

    def forward(self, local_obs, action):
        events, valid, overflow = self.events(local_obs, action)
        tokens = self.encode_events(events)
        return tokens, events, valid, overflow
