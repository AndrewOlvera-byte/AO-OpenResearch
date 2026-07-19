"""Discrete action-JEPA pretraining for the categorical state interface."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import FactorizedActionEventEncoder
from .config import DiscreteActionTokenizerConfig
from .schema import dense_actions_to_events


def _depth_logits(heads: nn.ModuleList, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the matching product-code head at each flattened code position."""
    depth = len(heads)
    pieces = []
    for offset, head in enumerate(heads):
        pieces.append(head(hidden[:, offset::depth]))
    out = hidden.new_empty(hidden.shape[0], hidden.shape[1], pieces[0].shape[-1])
    for offset, value in enumerate(pieces):
        out[:, offset::depth] = value
    return out


class DiscreteActionCodeRouter(nn.Module):
    """Current hard state codes + action events -> next-code prior logits."""

    def __init__(
        self,
        d_state: int,
        d_model: int,
        codebook_depth: int,
        codebook_size: int,
        n_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.state_in = nn.Linear(d_state, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.trunk = nn.Sequential(
            nn.Linear(d_model, int(mlp_ratio * d_model)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * d_model), d_model),
            nn.LayerNorm(d_model),
        )
        self.heads = nn.ModuleList(
            nn.Linear(d_model, codebook_size) for _ in range(codebook_depth)
        )

    def forward(self, state_tokens, action_tokens, valid):
        query = self.state_in(state_tokens)
        # ``MultiheadAttention`` returns NaNs when every key in a row is masked.
        # Empty joint-action rows are legal (for example before a unit can act),
        # so give attention one harmless sentinel key and then explicitly erase
        # its contribution.  The remaining query residual is the action-free
        # next-code prior for that row.
        empty = ~valid.any(dim=1)
        if empty.any():
            valid = valid.clone()
            action_tokens = action_tokens.clone()
            valid[empty, 0] = True
            action_tokens[empty, 0] = 0
        routed = self.attn(
            query,
            action_tokens,
            action_tokens,
            key_padding_mask=~valid,
            need_weights=False,
        )[0]
        routed = routed.masked_fill(empty[:, None, None], 0)
        hidden = self.trunk(self.norm(query + routed))
        return _depth_logits(self.heads, hidden)


class DiscreteActionTokenizerPretrainer(nn.Module):
    """Exact event codec plus forward/inverse categorical JEPA scaffolding."""

    def __init__(self, state_tokenizer, grid_hw=(16, 16), cfg=None):
        super().__init__()
        self.cfg = cfg or DiscreteActionTokenizerConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        d = int(self.cfg.d_model)
        self.action_encoder = FactorizedActionEventEncoder(
            d,
            self.grid_hw,
            self.cfg.max_unit_types,
            self.cfg.field_dim,
        )
        self.action_position = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        nn.init.normal_(self.action_position, std=0.02)
        self.router = DiscreteActionCodeRouter(
            state_tokenizer.d_latent,
            d,
            state_tokenizer.codebook_depth,
            state_tokenizer.codebook_size,
            self.cfg.n_heads,
            self.cfg.mlp_ratio,
            self.cfg.dropout,
        )

        self.inverse_before = nn.Linear(state_tokenizer.d_latent, d)
        self.inverse_after = nn.Linear(state_tokenizer.d_latent, d)
        self.inverse_delta = nn.Linear(state_tokenizer.d_latent, d)
        self.inverse_type = nn.Parameter(torch.zeros(3, d))
        self.inverse_queries = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        nn.init.normal_(self.inverse_queries, std=0.02)
        self.inverse_cross = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d,
            self.cfg.n_heads,
            int(self.cfg.mlp_ratio * d),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.inverse_transformer = nn.TransformerEncoder(
            layer, self.cfg.inverse_depth, nn.LayerNorm(d)
        )
        h, w = self.grid_hw
        self.event_heads = nn.ModuleDict(
            {
                "valid": nn.Linear(d, 1),
                "role": nn.Linear(d, 3),
                "action_type": nn.Linear(d, 6),
                "src_x": nn.Linear(d, w),
                "src_y": nn.Linear(d, h),
                "dst_x": nn.Linear(d, w + 2),
                "dst_y": nn.Linear(d, h + 2),
                "direction": nn.Linear(d, 6),
                "produced": nn.Linear(d, self.cfg.max_unit_types + 1),
                "attack": nn.Linear(d, 50),
            }
        )

    def action_events(self, state, action, opponent_action):
        return dense_actions_to_events(
            state,
            action,
            opponent_action,
            self.grid_hw,
            self.cfg.max_action_events,
        )

    def encode_events(self, events):
        return self.action_encoder(events) + self.action_position[: events.shape[-2]]

    def forward_logits(self, state_code_tokens, action_tokens, valid):
        return self.router(state_code_tokens, action_tokens, valid)

    def inverse_tokens(self, before_tokens, after_tokens):
        context = torch.cat(
            (
                self.inverse_before(before_tokens) + self.inverse_type[0],
                self.inverse_after(after_tokens) + self.inverse_type[1],
                self.inverse_delta(after_tokens - before_tokens) + self.inverse_type[2],
            ),
            dim=1,
        )
        query = self.inverse_queries[None].expand(before_tokens.shape[0], -1, -1)
        query = (
            query + self.inverse_cross(query, context, context, need_weights=False)[0]
        )
        return self.inverse_transformer(query)

    def decode_events(self, tokens):
        return {name: head(tokens) for name, head in self.event_heads.items()}


def _event_targets(events, model):
    role, _uid, sx, sy, typ, tx, ty, direction, produced, attack = events.unbind(-1)
    h, w = model.grid_hw
    return {
        "role": role.clamp(0, 2),
        "action_type": typ.clamp(0, 5),
        "src_x": sx.clamp(0, w - 1),
        "src_y": sy.clamp(0, h - 1),
        "dst_x": (tx + 1).clamp(0, w + 1),
        "dst_y": (ty + 1).clamp(0, h + 1),
        "direction": (direction + 1).clamp(0, 5),
        "produced": (produced + 1).clamp(0, model.cfg.max_unit_types),
        "attack": (attack + 1).clamp(0, 49),
    }


def _event_loss(model, tokens, events, valid, prefix):
    decoded = model.decode_events(tokens)
    valid_f = valid.to(tokens.dtype)
    denom = valid_f.sum().clamp_min(1.0)
    losses = [F.binary_cross_entropy_with_logits(decoded["valid"].squeeze(-1), valid_f)]
    exact = valid.clone()
    for name, target in _event_targets(events, model).items():
        logits = decoded[name]
        per = F.cross_entropy(
            logits.flatten(0, -2), target.flatten(), reduction="none"
        ).reshape_as(target)
        losses.append((per * valid_f).sum() / denom)
        exact &= logits.argmax(-1) == target
    return torch.stack(losses).mean(), {
        f"dajepa/{prefix}_valid_acc": ((decoded["valid"].squeeze(-1) >= 0) == valid)
        .float()
        .mean()
        .detach(),
        f"dajepa/{prefix}_exact_event": (exact.sum().float() / denom).detach(),
    }


def _balanced_code_ce(logits, target, current):
    per = F.cross_entropy(
        logits.flatten(0, -2), target.flatten(), reduction="none"
    ).reshape_as(target)
    changed = target != current
    unchanged = ~changed
    changed_loss = per[changed].mean() if changed.any() else per.new_zeros(())
    unchanged_loss = per[unchanged].mean() if unchanged.any() else per.new_zeros(())
    return changed_loss + unchanged_loss, changed_loss, unchanged_loss, changed


def _paired_margin(factual_logits, cf_logits, factual_target, cf_target, margin):
    effect = factual_target != cf_target
    if not effect.any():
        return factual_logits.new_zeros(()), effect
    lf = factual_logits.log_softmax(-1)
    lc = cf_logits.log_softmax(-1)
    yf = factual_target.unsqueeze(-1)
    yc = cf_target.unsqueeze(-1)
    factual_preference = lf.gather(-1, yf).squeeze(-1) - lf.gather(-1, yc).squeeze(-1)
    cf_preference = lc.gather(-1, yc).squeeze(-1) - lc.gather(-1, yf).squeeze(-1)
    loss = F.relu(float(margin) - factual_preference[effect]).mean()
    loss = loss + F.relu(float(margin) - cf_preference[effect]).mean()
    return loss, effect


def discrete_action_jepa_loss(
    model: DiscreteActionTokenizerPretrainer,
    state_tokenizer,
    batch: dict,
    *,
    reconstruction_coef: float = 1.0,
    inverse_coef: float = 1.0,
    forward_coef: float = 1.0,
    counterfactual_coef: float = 1.0,
    effect_margin_coef: float = 2.0,
    alignment_coef: float = 0.1,
    effect_margin: float = 1.0,
):
    """Categorical counterpart of the successful continuous action SSL run."""
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opponent = batch["opponent_action"].reshape(
        -1, *batch["opponent_action"].shape[-2:]
    )
    with torch.no_grad():
        c0 = state_tokenizer.encode_codes(state, glob)
        c1 = state_tokenizer.encode_codes(nxt, nglob)
        z0 = state_tokenizer.embed_code_tokens(c0)
        z1 = state_tokenizer.embed_code_tokens(c1)
    flat0, flat1 = c0.flatten(1), c1.flatten(1)
    events, valid, overflow = model.action_events(state, action, opponent)
    action_tokens = model.encode_events(events)
    reconstruction, metrics = _event_loss(model, action_tokens, events, valid, "recon")
    inverse_tokens = model.inverse_tokens(z0, z1)
    inverse, inverse_metrics = _event_loss(
        model, inverse_tokens, events, valid, "inverse"
    )
    metrics.update(inverse_metrics)
    valid_f = valid.to(action_tokens.dtype)
    alignment = (
        (inverse_tokens - action_tokens.detach()).pow(2).mean(-1) * valid_f
    ).sum() / valid_f.sum().clamp_min(1.0)

    factual_logits = model.forward_logits(z0, action_tokens, valid)
    forward, changed_ce, unchanged_ce, changed = _balanced_code_ce(
        factual_logits, flat1, flat0
    )
    counterfactual = forward.new_zeros(())
    effect_loss = forward.new_zeros(())
    effect = torch.zeros_like(flat1, dtype=torch.bool)
    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    if cf_valid.any():
        idx = torch.nonzero(cf_valid, as_tuple=False).flatten()
        raw_cf_state = batch["counterfactual_next_state"].reshape(
            -1, *batch["counterfactual_next_state"].shape[-2:]
        )
        raw_cf_glob = batch["counterfactual_next_globals"].reshape(
            -1, batch["counterfactual_next_globals"].shape[-1]
        )
        aligned_state = torch.where(cf_valid[:, None, None], raw_cf_state, nxt)
        aligned_glob = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
        with torch.no_grad():
            ccf_all = state_tokenizer.encode_codes(aligned_state, aligned_glob)
        ccf = ccf_all[idx].flatten(1)
        cf_action = batch["counterfactual_action"].reshape(
            -1, *batch["counterfactual_action"].shape[-2:]
        )[idx]
        cf_opp = batch["counterfactual_opponent_action"].reshape(
            -1, *batch["counterfactual_opponent_action"].shape[-2:]
        )[idx]
        cf_events, cf_event_valid, _ = model.action_events(
            state[idx], cf_action, cf_opp
        )
        cf_tokens = model.encode_events(cf_events)
        cf_logits = model.forward_logits(z0[idx], cf_tokens, cf_event_valid)
        counterfactual = _balanced_code_ce(cf_logits, ccf, flat0[idx])[0]
        effect_loss, local_effect = _paired_margin(
            factual_logits[idx], cf_logits, flat1[idx], ccf, effect_margin
        )
        effect[idx] = local_effect

    total = (
        float(reconstruction_coef) * reconstruction
        + float(inverse_coef) * inverse
        + float(forward_coef) * forward
        + float(counterfactual_coef) * counterfactual
        + float(effect_margin_coef) * effect_loss
        + float(alignment_coef) * alignment
    )
    with torch.no_grad():
        pred = factual_logits.argmax(-1)
        changed_correct = (
            (pred[changed] == flat1[changed]).float().mean()
            if changed.any()
            else total.new_ones(())
        )
        effect_fraction = effect.float().mean()
    metrics.update(
        {
            "dajepa/total": total.detach(),
            "dajepa/reconstruction": reconstruction.detach(),
            "dajepa/inverse": inverse.detach(),
            "dajepa/forward": forward.detach(),
            "dajepa/forward_changed": changed_ce.detach(),
            "dajepa/forward_unchanged": unchanged_ce.detach(),
            "dajepa/counterfactual": counterfactual.detach(),
            "dajepa/effect_margin": effect_loss.detach(),
            "dajepa/alignment": alignment.detach(),
            "dajepa/code_acc": (pred == flat1).float().mean(),
            "dajepa/changed_code_acc": changed_correct,
            "dajepa/effect_code_fraction": effect_fraction,
            "dajepa/counterfactual_fraction": cf_valid.float().mean(),
            "dajepa/action_overflow": overflow.float().mean(),
        }
    )
    return total, metrics
