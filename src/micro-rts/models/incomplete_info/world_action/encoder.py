from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .attention import CrossBlock, SDPCrossAttention, TransformerBlock, sinusoidal_positions
from .config import PredictiveBeliefConfig


BRANCHES = ("self", "opponent", "static", "interaction")


def split_branches(value, sizes):
    return dict(zip(BRANCHES, value.split(tuple(sizes), dim=-2)))


def masked_action_pool(tokens, valid):
    weight = valid[..., None].to(tokens.dtype)
    return (tokens * weight).sum(-2) / weight.sum(-2).clamp_min(1.0)


class WorldActionBeliefEncoder(nn.Module):
    """Fog-history encoder whose state is factorized by causal planning role."""

    def __init__(self, spatial_dim=320, action_dim=320, cfg=None):
        super().__init__()
        self.cfg = cfg or PredictiveBeliefConfig()
        d = self.cfg.d_model
        self.spatial_in = nn.Linear(spatial_dim, d)
        self.action_in = nn.Linear(action_dim, d)
        self.visibility_in = nn.Linear(1, d)
        self.modality = nn.Parameter(torch.zeros(2, d))
        self.null_action = nn.Parameter(torch.zeros(d))
        self.branch_queries = nn.Parameter(torch.zeros(self.cfg.n_tokens, d))
        self.branch_type = nn.Parameter(torch.zeros(4, d))
        nn.init.normal_(self.modality, std=0.02)
        nn.init.normal_(self.null_action, std=0.02)
        nn.init.normal_(self.branch_queries, std=0.02)
        nn.init.normal_(self.branch_type, std=0.02)
        frame_kw = dict(
            dim=d,
            heads=self.cfg.n_heads,
            ratio=self.cfg.mlp_ratio,
            dropout=self.cfg.dropout,
        )
        self.frame_cross = CrossBlock(**frame_kw)
        self.frame_blocks = nn.ModuleList(
            TransformerBlock(**frame_kw) for _ in range(self.cfg.frame_depth)
        )
        self.temporal_blocks = nn.ModuleList(
            TransformerBlock(**frame_kw) for _ in range(self.cfg.temporal_depth)
        )
        self.mixer_blocks = nn.ModuleList(
            TransformerBlock(**frame_kw) for _ in range(self.cfg.mixer_depth)
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.cfg.d_latent)

    def _typed_queries(self, count, device, dtype):
        query = self.branch_queries
        chunks = []
        for index, part in enumerate(query.split(self.cfg.branch_sizes, dim=0)):
            chunks.append(part + self.branch_type[index])
        return torch.cat(chunks, dim=0).to(device=device, dtype=dtype)[None].expand(
            count, -1, -1
        )

    def forward(self, spatial, visibility, action_tokens, action_valid):
        if spatial.shape[1] > self.cfg.context_length:
            spatial = spatial[:, -self.cfg.context_length :]
            visibility = visibility[:, -self.cfg.context_length :]
            action_tokens = action_tokens[:, -self.cfg.context_length :]
            action_valid = action_valid[:, -self.cfg.context_length :]
        b, t, ns = spatial.shape[:3]
        na = action_tokens.shape[-2]
        spatial_memory = self.spatial_in(spatial) + self.visibility_in(
            visibility.to(spatial.dtype)
        )
        spatial_memory = spatial_memory + self.modality[0]
        spatial_memory = spatial_memory + sinusoidal_positions(
            ns, self.cfg.d_model, spatial.device, spatial_memory.dtype
        ).view(1, 1, ns, -1)
        action_memory = self.action_in(action_tokens)
        action_memory = torch.where(
            action_valid[..., None],
            action_memory + self.modality[1],
            self.null_action.view(1, 1, 1, -1),
        )
        action_memory = action_memory + sinusoidal_positions(
            na, self.cfg.d_model, action_tokens.device, action_memory.dtype
        ).view(1, 1, na, -1)
        memory = torch.cat((spatial_memory, action_memory), dim=2).reshape(
            b * t, ns + na, self.cfg.d_model
        )
        value = self.frame_cross(
            self._typed_queries(b * t, memory.device, memory.dtype), memory
        )
        for block in self.frame_blocks:
            value = block(value)
        value = value.reshape(b, t, self.cfg.n_tokens, self.cfg.d_model)

        # Each semantic slot receives a causal flash-attention temporal path.
        temporal = value.transpose(1, 2).reshape(
            b * self.cfg.n_tokens, t, self.cfg.d_model
        )
        temporal = temporal + sinusoidal_positions(
            t, self.cfg.d_model, temporal.device, temporal.dtype
        )[None]
        for block in self.temporal_blocks:
            temporal = block(temporal, causal=True)
        value = temporal.reshape(b, self.cfg.n_tokens, t, self.cfg.d_model).transpose(1, 2)

        mixed = value.reshape(b * t, self.cfg.n_tokens, self.cfg.d_model)
        for block in self.mixer_blocks:
            mixed = block(mixed)
        latent = self.out(self.norm(mixed)).reshape(
            b, t, self.cfg.n_tokens, self.cfg.d_latent
        )
        return {"tokens": latent, **split_branches(latent, self.cfg.branch_sizes)}


class MultiHorizonPredictor(nn.Module):
    def __init__(self, action_dim, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.latent_in = nn.Linear(cfg.d_latent, d)
        self.action_in = nn.Linear(action_dim, d)
        self.horizon = nn.Embedding(len(cfg.horizons), d)
        self.blocks = nn.ModuleList(
            TransformerBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.predictor_depth)
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.d_latent)

    def one(self, tokens, action_summary, horizon_index):
        value = self.latent_in(tokens)
        value = value + self.action_in(action_summary)[..., None, :]
        value = value + self.horizon.weight[horizon_index]
        lead = value.shape[:-2]
        value = value.reshape(-1, value.shape[-2], value.shape[-1])
        for block in self.blocks:
            value = block(value)
        return self.out(self.norm(value)).reshape(*lead, self.cfg.n_tokens, self.cfg.d_latent)

    def forward(self, tokens, action_pool):
        outputs = {}
        for index, horizon in enumerate(self.cfg.horizons):
            if tokens.shape[1] <= horizon:
                continue
            windows = action_pool.unfold(1, horizon, 1).mean(-1)
            # unfold includes a final window with no corresponding future belief.
            windows = windows[:, : tokens.shape[1] - horizon]
            outputs[horizon] = self.one(
                tokens[:, : tokens.shape[1] - horizon], windows, index
            )
        return outputs


class TokenQueryHead(nn.Module):
    def __init__(self, n_queries, in_dim, out_dim, d_model, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(n_queries, d_model))
        nn.init.normal_(self.queries, std=0.02)
        self.memory_in = nn.Linear(in_dim, d_model)
        self.cross = SDPCrossAttention(d_model, heads)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, out_dim)

    def forward(self, memory):
        lead = memory.shape[:-2]
        flat = self.memory_in(memory.reshape(-1, *memory.shape[-2:]))
        query = self.queries[None].expand(flat.shape[0], -1, -1)
        value = query + self.cross(query, flat)
        return self.out(self.norm(value)).reshape(*lead, self.queries.shape[0], -1)


class PredictiveBeliefPretrainer(nn.Module):
    """New planning encoder plus EMA target and transition/value auxiliaries."""

    def __init__(
        self,
        ego_tokenizer,
        self_action_tokenizer,
        opponent_tokenizer,
        cfg=None,
    ):
        super().__init__()
        self.cfg = cfg or PredictiveBeliefConfig()
        self.ego_tokenizer = ego_tokenizer.requires_grad_(False).eval()
        self.self_action_tokenizer = self_action_tokenizer.requires_grad_(False).eval()
        self.opponent_tokenizer = opponent_tokenizer.requires_grad_(False).eval()
        self.encoder = WorldActionBeliefEncoder(
            ego_tokenizer.cfg.d_latent,
            self_action_tokenizer.cfg.d_latent,
            self.cfg,
        )
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False).eval()
        self.predictor = MultiHorizonPredictor(
            self_action_tokenizer.cfg.d_latent, self.cfg
        )
        dlat = self.cfg.d_latent
        self.self_inverse = nn.Sequential(
            nn.LayerNorm(2 * dlat),
            nn.Linear(2 * dlat, self.cfg.d_model),
            nn.SiLU(),
            nn.Linear(self.cfg.d_model, dlat),
        )
        self.opponent_plan_head = TokenQueryHead(
            opponent_tokenizer.cfg.n_plan_tokens,
            dlat,
            opponent_tokenizer.cfg.d_latent,
            self.cfg.d_model,
            self.cfg.n_heads,
        )
        self.event_head = TokenQueryHead(
            ego_tokenizer.n_spatial,
            dlat,
            self.cfg.event_dim,
            self.cfg.d_model,
            self.cfg.n_heads,
        )
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(dlat),
            nn.Linear(dlat, self.cfg.d_model),
            nn.SiLU(),
            nn.Linear(self.cfg.d_model, 3),
        )

    def train(self, mode=True):
        super().train(mode)
        self.ego_tokenizer.eval()
        self.self_action_tokenizer.eval()
        self.opponent_tokenizer.eval()
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        decay = float(self.cfg.ema_decay)
        for target, source in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    @torch.no_grad()
    def tokenize(self, batch, action_key="action"):
        spatial, _, visibility = self.ego_tokenizer.encode(
            batch["local_obs"], batch["local_visibility"]
        )
        current_action, _, current_valid, _ = self.self_action_tokenizer(
            batch["local_obs"], batch[action_key]
        )
        # The belief at t may consume actions only through t-1.  The unshifted
        # tokens are returned separately as interventions for future prediction.
        history_action = torch.zeros_like(current_action)
        history_valid = torch.zeros_like(current_valid)
        history_action[:, 1:] = current_action[:, :-1]
        history_valid[:, 1:] = current_valid[:, :-1]
        is_first = batch.get("is_first")
        if is_first is not None:
            history_valid = history_valid & ~is_first[..., None].bool()
        return (
            spatial,
            visibility,
            history_action,
            history_valid,
            current_action,
            current_valid,
        )

    def encode_tokenized(self, spatial, visibility, action, valid):
        online = self.encoder(spatial, visibility, action, valid)
        with torch.no_grad():
            target = self.target_encoder(spatial, visibility, action, valid)
        return online, target, masked_action_pool(action, valid)

    def forward(self, batch):
        (
            spatial,
            visibility,
            history_action,
            history_valid,
            action,
            valid,
        ) = self.tokenize(batch)
        online, target, action_pool = self.encode_tokenized(
            spatial, visibility, history_action, history_valid
        )
        action_pool = masked_action_pool(action, valid)
        predictions = self.predictor(online["tokens"], action_pool)
        return {
            "online": online,
            "target": target,
            "predictions": predictions,
            "action_tokens": action,
            "action_valid": valid,
            "action_pool": action_pool,
            "spatial": spatial,
            "visibility": visibility,
        }

    def predict_counterfactual(self, encoded, batch):
        with torch.no_grad():
            action, _, valid, _ = self.self_action_tokenizer(
                batch["local_obs"], batch["counterfactual_action"]
            )
            pool = masked_action_pool(action, valid)
        return self.predictor.one(encoded["online"]["tokens"][:, :-1], pool[:, :-1], 0)

    def scalar_predictions(self, tokens):
        return self.scalar_head(tokens.mean(-2))

    def inverse_action(self, source_self, future_self):
        return self.self_inverse(
            torch.cat((source_self.mean(-2), future_self.mean(-2)), dim=-1)
        )
