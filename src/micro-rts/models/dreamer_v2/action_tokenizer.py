"""Pretrainable multi-event action representation for structured MicroRTS.

The action tokenizer keeps one embedding per issued event.  Pretraining combines
three complementary signals:

* exact event-field autoencoding (the representation is auditable/invertible),
* forward latent-delta prediction including cloned counterfactual effects, and
* inverse action inference from the frozen state-tokenizer transition.

Only :class:`FactorizedActionEventEncoder` is consumed by the downstream world
model; the forward/inverse heads are disposable SSL training scaffolding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ActionTokenizerConfig
from .schema import dense_actions_to_events


class FactorizedActionEventEncoder(nn.Module):
    """Sparse action records -> one embedding per event without early summing."""

    def __init__(
        self,
        d_model: int,
        grid_hw=(16, 16),
        max_unit_types: int = 16,
        field_dim: int = 64,
    ):
        super().__init__()
        h, w = map(int, grid_hw)
        self.h, self.w = h, w
        f = int(field_dim)
        self.role = nn.Embedding(3, f)
        self.atype = nn.Embedding(6, f)
        self.direction = nn.Embedding(6, f)  # -1..4 shifted
        self.produced = nn.Embedding(max_unit_types + 1, f)
        self.src_x = nn.Embedding(w, f)
        self.src_y = nn.Embedding(h, f)
        self.dst_x = nn.Embedding(w + 2, f)  # -1 sentinel plus board/edge
        self.dst_y = nn.Embedding(h + 2, f)
        self.attack = nn.Sequential(nn.Linear(1, f), nn.SiLU(), nn.Linear(f, f))
        self.project = nn.Sequential(
            nn.Linear(9 * f, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        role, _uid, sx, sy, typ, tx, ty, direction, produced, attack = events.unbind(-1)
        fields = (
            self.role(role.clamp(0, 2)),
            self.atype(typ.clamp(0, 5)),
            self.direction((direction + 1).clamp(0, 5)),
            self.produced((produced + 1).clamp(0, self.produced.num_embeddings - 1)),
            self.src_x(sx.clamp(0, self.w - 1)),
            self.src_y(sy.clamp(0, self.h - 1)),
            self.dst_x((tx + 1).clamp(0, self.w + 1)),
            self.dst_y((ty + 1).clamp(0, self.h + 1)),
            self.attack(attack.float().unsqueeze(-1) / 49.0),
        )
        # Raw JVM unit IDs are deliberately excluded. Role + source coordinate
        # is the stable binding in the single-occupancy canonical state.
        return self.project(torch.cat(fields, dim=-1))


class ActionTokenizerPretrainer(nn.Module):
    """Multi-token action encoder plus disposable forward/inverse SSL heads."""

    def __init__(
        self,
        n_state_tokens: int,
        d_latent: int,
        grid_hw=(16, 16),
        cfg: ActionTokenizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or ActionTokenizerConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        self.n_state_tokens = int(n_state_tokens)
        self.d_latent = int(d_latent)
        d = self.cfg.d_model
        self.action_encoder = FactorizedActionEventEncoder(
            d,
            self.grid_hw,
            self.cfg.max_unit_types,
            self.cfg.field_dim,
        )
        self.action_position = nn.Parameter(
            torch.zeros(self.cfg.max_action_events, d)
        )
        nn.init.normal_(self.action_position, std=0.02)

        # Forward SSL: every state token queries the issued action-event set and
        # predicts its normalized one-step latent delta.
        self.forward_state = nn.Linear(d_latent, d)
        self.forward_attn = nn.MultiheadAttention(
            d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
        )
        self.forward_norm = nn.LayerNorm(d)
        self.forward_head = nn.Sequential(
            nn.Linear(d, int(self.cfg.mlp_ratio * d)),
            nn.GELU(),
            nn.Linear(int(self.cfg.mlp_ratio * d), d_latent),
        )

        # Inverse SSL: learned event-slot queries read before/after/delta state
        # tokens and infer the issued event embeddings and fields.
        self.inverse_before = nn.Linear(d_latent, d)
        self.inverse_after = nn.Linear(d_latent, d)
        self.inverse_delta = nn.Linear(d_latent, d)
        self.inverse_type = nn.Parameter(torch.zeros(3, d))
        self.inverse_queries = nn.Parameter(
            torch.zeros(self.cfg.max_action_events, d)
        )
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
                "attack": nn.Linear(d, 50),  # -1 sentinel + offsets 0..48
            }
        )
        self.register_buffer("latent_mean", torch.zeros(1, 1, d_latent))
        self.register_buffer("latent_std", torch.ones(1, 1, d_latent))

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.latent_mean.copy_(mean.reshape(1, 1, -1))
        self.latent_std.copy_(std.reshape(1, 1, -1).clamp_min(1e-4))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.latent_mean) / self.latent_std

    def encode_events(self, events: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(events) + self.action_position[: events.shape[-2]]

    def decode_events(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(tokens) for name, head in self.event_heads.items()}

    def forward_delta(
        self, state_z: torch.Tensor, action_tokens: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        q = self.forward_state(state_z)
        routed = self.forward_attn(
            q,
            action_tokens,
            action_tokens,
            key_padding_mask=~valid,
            need_weights=False,
        )[0]
        return self.forward_head(self.forward_norm(q + routed))

    def inverse_tokens(self, before_z: torch.Tensor, after_z: torch.Tensor) -> torch.Tensor:
        context = torch.cat(
            (
                self.inverse_before(before_z) + self.inverse_type[0],
                self.inverse_after(after_z) + self.inverse_type[1],
                self.inverse_delta(after_z - before_z) + self.inverse_type[2],
            ),
            dim=1,
        )
        q = self.inverse_queries[None].expand(before_z.shape[0], -1, -1)
        q = q + self.inverse_cross(q, context, context, need_weights=False)[0]
        return self.inverse_transformer(q)

    def action_events(self, state, action, opponent_action):
        return dense_actions_to_events(
            state,
            action,
            opponent_action,
            self.grid_hw,
            self.cfg.max_action_events,
        )


def _event_targets(events: torch.Tensor, model: ActionTokenizerPretrainer):
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
        "produced": (produced + 1).clamp(
            0, model.cfg.max_unit_types
        ),
        "attack": (attack + 1).clamp(0, 49),
    }


def _event_decode_loss(model, tokens, events, valid, prefix):
    decoded = model.decode_events(tokens)
    valid_f = valid.to(tokens.dtype)
    loss_valid = F.binary_cross_entropy_with_logits(
        decoded["valid"].squeeze(-1), valid_f
    )
    targets = _event_targets(events, model)
    losses = [loss_valid]
    correct = {}
    denom = valid_f.sum().clamp_min(1.0)
    for name, target in targets.items():
        logits = decoded[name]
        per = F.cross_entropy(
            logits.flatten(0, -2), target.flatten(), reduction="none"
        ).reshape_as(target)
        losses.append((per * valid_f).sum() / denom)
        correct[name] = (((logits.argmax(-1) == target) & valid).sum() / denom).detach()
    total = torch.stack(losses).mean()
    metrics = {
        f"action_tok/{prefix}_valid_acc": (
            ((decoded["valid"].squeeze(-1) >= 0) == valid).float().mean()
        ).detach(),
        f"action_tok/{prefix}_type_acc": correct["action_type"],
        f"action_tok/{prefix}_source_acc": (
            (correct["src_x"] + correct["src_y"]) * 0.5
        ),
        f"action_tok/{prefix}_target_acc": (
            (correct["dst_x"] + correct["dst_y"]) * 0.5
        ),
    }
    return total, metrics


def _state_token_weights(tokenizer, states, delta, changed_boost, threshold, padding):
    batch = delta.shape[0]
    valid = torch.ones(
        batch, tokenizer.n_tokens, dtype=torch.bool, device=delta.device
    )
    counts = torch.stack([s[..., 1].bool().sum(-1) for s in states]).amax(0)
    slots = torch.arange(tokenizer.n_entity, device=delta.device)[None]
    entity_valid = slots < counts[:, None].clamp_max(tokenizer.n_entity)
    lo = tokenizer.n_spatial
    valid[:, lo : lo + tokenizer.n_entity] = entity_valid
    changed = delta.pow(2).mean(-1) > float(threshold)
    weights = torch.where(
        valid,
        torch.ones_like(delta[..., 0]),
        torch.full_like(delta[..., 0], float(padding)),
    )
    return weights * (1.0 + (float(changed_boost) - 1.0) * changed.to(weights.dtype))


def _weighted_mse(pred, target, weights):
    per = (pred - target).pow(2).mean(-1)
    return (per * weights).sum() / weights.sum().clamp_min(1.0)


def action_tokenizer_ssl_loss(
    model: ActionTokenizerPretrainer,
    state_tokenizer,
    batch: dict,
    *,
    reconstruction_coef: float = 1.0,
    inverse_coef: float = 1.0,
    forward_coef: float = 1.0,
    paired_effect_coef: float = 4.0,
    alignment_coef: float = 0.1,
    changed_token_boost: float = 8.0,
    change_threshold: float = 1e-3,
    padding_token_weight: float = 0.05,
):
    """Dual forward/inverse SSL objective for the action representation."""
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opponent = batch["opponent_action"].reshape(
        -1, *batch["opponent_action"].shape[-2:]
    )
    with torch.no_grad():
        z0 = model.normalize(state_tokenizer.encode(state, glob))
        z1 = model.normalize(state_tokenizer.encode(nxt, nglob))

    events, valid, overflow = model.action_events(state, action, opponent)
    action_tokens = model.encode_events(events)
    reconstruction, metrics = _event_decode_loss(
        model, action_tokens, events, valid, "recon"
    )

    inverse_tokens = model.inverse_tokens(z0, z1)
    inverse, inverse_metrics = _event_decode_loss(
        model, inverse_tokens, events, valid, "inverse"
    )
    metrics.update(inverse_metrics)
    valid_f = valid.to(action_tokens.dtype)
    alignment = (
        (inverse_tokens - action_tokens.detach()).pow(2).mean(-1) * valid_f
    ).sum() / valid_f.sum().clamp_min(1.0)

    target_delta = z1 - z0
    predicted_delta = model.forward_delta(z0, action_tokens, valid)
    forward_weights = _state_token_weights(
        state_tokenizer,
        (state, nxt),
        target_delta,
        changed_token_boost,
        change_threshold,
        padding_token_weight,
    )
    forward = _weighted_mse(predicted_delta, target_delta, forward_weights)

    paired = forward.new_zeros(())
    effect_cosine = forward.new_zeros(())
    effect_norm_ratio = forward.new_zeros(())
    effect_norm_ratio_aggregate = forward.new_zeros(())
    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    if cf_valid.any():
        idx = torch.nonzero(cf_valid, as_tuple=False).flatten()
        raw_cf_state = batch["counterfactual_next_state"].reshape(
            -1, *batch["counterfactual_next_state"].shape[-2:]
        )
        raw_cf_glob = batch["counterfactual_next_globals"].reshape(
            -1, batch["counterfactual_next_globals"].shape[-1]
        )
        cf_state_all = torch.where(
            cf_valid[:, None, None], raw_cf_state, nxt
        )
        cf_glob_all = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
        cf_state = cf_state_all[idx]
        cf_action = batch["counterfactual_action"].reshape(
            -1, *batch["counterfactual_action"].shape[-2:]
        )[idx]
        cf_opp = batch["counterfactual_opponent_action"].reshape(
            -1, *batch["counterfactual_opponent_action"].shape[-2:]
        )[idx]
        with torch.no_grad():
            zcf = model.normalize(
                state_tokenizer.encode(cf_state_all, cf_glob_all)
            )[idx]
        cf_events, cf_event_valid, _ = model.action_events(
            state[idx], cf_action, cf_opp
        )
        cf_tokens = model.encode_events(cf_events)
        cf_delta = model.forward_delta(z0[idx], cf_tokens, cf_event_valid)
        target_effect = zcf - z1[idx]
        predicted_effect = cf_delta - predicted_delta[idx]
        effect_weights = _state_token_weights(
            state_tokenizer,
            (nxt[idx], cf_state),
            target_effect,
            changed_token_boost,
            change_threshold,
            padding_token_weight,
        )
        paired = _weighted_mse(predicted_effect, target_effect, effect_weights)
        flat_target = target_effect.flatten(1)
        flat_pred = predicted_effect.flatten(1)
        has_effect = flat_target.norm(dim=1) > 1e-8
        if has_effect.any():
            predicted_norm = flat_pred[has_effect].norm(dim=1)
            target_norm = flat_target[has_effect].norm(dim=1)
            effect_cosine = F.cosine_similarity(
                flat_pred[has_effect], flat_target[has_effect], dim=1
            ).mean()
            effect_norm_ratio = (predicted_norm / target_norm.clamp_min(1e-8)).mean()
            effect_norm_ratio_aggregate = (
                predicted_norm.sum() / target_norm.sum().clamp_min(1e-8)
            )

    total = (
        float(reconstruction_coef) * reconstruction
        + float(inverse_coef) * inverse
        + float(forward_coef) * forward
        + float(paired_effect_coef) * paired
        + float(alignment_coef) * alignment
    )
    metrics.update(
        {
            "action_tok/total": total.detach(),
            "action_tok/reconstruction": reconstruction.detach(),
            "action_tok/inverse": inverse.detach(),
            "action_tok/forward": forward.detach(),
            "action_tok/paired_effect": paired.detach(),
            "action_tok/alignment": alignment.detach(),
            "action_tok/effect_cosine": effect_cosine.detach(),
            "action_tok/effect_norm_ratio": effect_norm_ratio.detach(),
            "action_tok/effect_norm_ratio_aggregate": (
                effect_norm_ratio_aggregate.detach()
            ),
            "action_tok/counterfactual_fraction": cf_valid.float().mean().detach(),
            "action_tok/overflow": overflow.float().mean().detach(),
            "action_tok/token_rms": action_tokens.detach().float().pow(2).mean().sqrt(),
        }
    )
    return total, metrics
