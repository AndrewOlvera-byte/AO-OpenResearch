"""Action-conditioned temporal JEPA for the structured observation tokenizer.

The online tokenizer remains a complete semantic autoencoder.  A slow EMA copy
defines next-state targets while a disposable sparse-action router predicts the
factual and cloned-counterfactual latent transitions.  Downstream checkpoints
extract only ``tokenizer.*``; the target encoder and predictor never become part
of the deployed world model.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import FactorizedActionEventEncoder
from .config import StructuredTokenizerConfig, TemporalJEPAConfig
from .schema import dense_actions_to_events
from .tokenizer import StructuredTokenizer, structured_reconstruction_loss


def _normalized(z: torch.Tensor) -> torch.Tensor:
    """Remove the latent scale degree of freedom without mixing token types."""
    return F.layer_norm(z.float(), (z.shape[-1],)).to(z.dtype)


class TemporalActionPredictor(nn.Module):
    """Sparse joint actions routed into one predicted delta per state token."""

    def __init__(self, n_tokens, d_latent, grid_hw, cfg: TemporalJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.grid_hw = tuple(map(int, grid_hw))
        d = cfg.d_model
        self.action_encoder = FactorizedActionEventEncoder(
            d, self.grid_hw, cfg.max_unit_types, cfg.field_dim
        )
        self.action_position = nn.Parameter(torch.zeros(cfg.max_action_events, d))
        nn.init.normal_(self.action_position, std=0.02)
        self.forward_state = nn.Linear(d_latent, d)
        self.forward_attn = nn.MultiheadAttention(
            d, cfg.n_heads, cfg.dropout, batch_first=True
        )
        self.forward_norm = nn.LayerNorm(d)
        hidden = int(cfg.mlp_ratio * d)
        self.forward_head = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d_latent)
        )
        self.n_tokens = int(n_tokens)

    def events(self, state, action, opponent_action):
        return dense_actions_to_events(
            state,
            action,
            opponent_action,
            self.grid_hw,
            self.cfg.max_action_events,
        )

    def forward(self, state_z, events, valid):
        tokens = self.action_encoder(events)
        tokens = tokens + self.action_position[: tokens.shape[-2]]
        # The schema retains an explicit no-action sentinel, but keep this path
        # safe for synthetic/audit batches that contain no valid event at all.
        empty = ~valid.any(1)
        if empty.any():
            tokens = tokens.clone()
            valid = valid.clone()
            tokens[empty, 0] = 0
            valid[empty, 0] = True
        query = self.forward_state(state_z)
        routed = self.forward_attn(
            query, tokens, tokens, key_padding_mask=~valid, need_weights=False
        )[0]
        return self.forward_head(self.forward_norm(query + routed))


class TemporalJEPATokenizerPretrainer(nn.Module):
    """Online tokenizer, EMA target tokenizer, and disposable JEPA predictor."""

    def __init__(
        self,
        grid_hw=(16, 16),
        tokenizer_cfg: StructuredTokenizerConfig | None = None,
        jepa_cfg: TemporalJEPAConfig | None = None,
    ):
        super().__init__()
        self.tokenizer = StructuredTokenizer(grid_hw, tokenizer_cfg)
        self.target_tokenizer = copy.deepcopy(self.tokenizer)
        self.target_tokenizer.requires_grad_(False)
        self.target_tokenizer.eval()
        self.jepa_cfg = jepa_cfg or TemporalJEPAConfig()
        self.predictor = TemporalActionPredictor(
            self.tokenizer.n_tokens,
            self.tokenizer.d_latent,
            grid_hw,
            self.jepa_cfg,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_tokenizer.eval()
        return self

    @torch.no_grad()
    def update_target(self, decay: float | None = None):
        decay = self.jepa_cfg.ema_decay if decay is None else float(decay)
        online = self.tokenizer.state_dict()
        target = self.target_tokenizer.state_dict()
        for name, value in target.items():
            source = online[name]
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)


def structured_tokenizer_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Return deployable tokenizer weights from base-AE or JEPA checkpoints."""
    state = checkpoint["model"]
    prefix = "tokenizer."
    extracted = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    return extracted or state


def _token_weights(tokenizer, states, target_delta, changed_boost, threshold, padding):
    batch = target_delta.shape[0]
    valid = torch.ones(
        batch, tokenizer.n_tokens, dtype=torch.bool, device=target_delta.device
    )
    counts = torch.stack([state[..., 1].bool().sum(-1) for state in states]).amax(0)
    entity = torch.arange(tokenizer.n_entity, device=target_delta.device)[None]
    lo = tokenizer.n_spatial
    valid[:, lo : lo + tokenizer.n_entity] = entity < counts[:, None].clamp_max(
        tokenizer.n_entity
    )
    changed = target_delta.float().pow(2).mean(-1) > float(threshold)
    changed &= valid
    weights = torch.where(
        valid,
        torch.ones_like(target_delta[..., 0]),
        torch.full_like(target_delta[..., 0], float(padding)),
    )
    weights *= 1.0 + (float(changed_boost) - 1.0) * changed.to(weights.dtype)
    return weights, valid, changed


def _weighted_mse(prediction, target, weights):
    per_token = (prediction.float() - target.float()).pow(2).mean(-1)
    return (per_token * weights.float()).sum() / weights.sum().clamp_min(1.0)


def _changed_cell_weights(source, target, boost):
    """Weight exact next-state grounding toward cells changed by the action."""
    learned_fields = [index for index in range(source.shape[-1]) if index != 2]
    changed = (source[..., learned_fields] != target[..., learned_fields]).any(-1)
    return 1.0 + (float(boost) - 1.0) * changed.float()


def _grounding_metrics(metrics, branch):
    names = (
        "exact_cell",
        "exact_frame",
        "exact_globals",
        "exact_roundtrip",
        "exact_occupied_cell",
        "exact_assigned_cell",
        "mechanics_score",
    )
    return {
        f"tok/jepa_{branch}_{name}": metrics[f"tok/{name}"]
        for name in names
    }


def structured_temporal_jepa_loss(
    model: TemporalJEPATokenizerPretrainer,
    batch: dict,
    *,
    reconstruction_coef: float = 1.0,
    factual_coef: float = 1.0,
    counterfactual_coef: float = 1.0,
    effect_coef: float = 2.0,
    factual_grounding_coef: float = 0.0,
    counterfactual_grounding_coef: float = 0.0,
    changed_token_boost: float = 8.0,
    change_threshold: float = 1.0e-3,
    padding_token_weight: float = 0.05,
):
    """Semantic reconstruction plus factual/counterfactual one-step JEPA."""
    if (
        factual_grounding_coef or counterfactual_grounding_coef
    ) and not model.jepa_cfg.raw_prediction:
        raise ValueError("exact JEPA grounding requires temporal_jepa.raw_prediction")
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opponent = batch["opponent_action"].reshape(
        -1, *batch["opponent_action"].shape[-2:]
    )

    decoded, raw_z0 = model.tokenizer(state, glob)
    reconstruction, recon_metrics = structured_reconstruction_loss(
        model.tokenizer,
        decoded,
        state,
        glob,
        (
            batch["obs"].reshape(-1, *batch["obs"].shape[-3:])
            if "obs" in batch
            else None
        ),
        (
            batch["mask"].reshape(-1, *batch["mask"].shape[-2:])
            if "mask" in batch
            else None
        ),
    )
    z0 = _normalized(raw_z0)
    with torch.no_grad():
        target_z0 = _normalized(model.target_tokenizer.encode(state, glob))
        target_z1 = _normalized(model.target_tokenizer.encode(nxt, nglob))

    events, event_valid, overflow = model.predictor.events(
        state, action, opponent
    )
    factual_delta = model.predictor(z0, events, event_valid)
    if model.jepa_cfg.raw_prediction:
        factual_raw = raw_z0 + factual_delta
        factual_pred = _normalized(factual_raw)
    else:
        factual_raw = None
        factual_pred = z0 + factual_delta
    target_delta = target_z1 - target_z0
    factual_weights, valid_tokens, changed = _token_weights(
        model.tokenizer,
        (state, nxt),
        target_delta,
        changed_token_boost,
        change_threshold,
        padding_token_weight,
    )
    factual = _weighted_mse(factual_pred, target_z1, factual_weights)

    factual_grounding = factual.new_zeros(())
    grounding_metrics = {}
    if factual_grounding_coef:
        factual_decoded = model.tokenizer.decode(factual_raw)
        factual_grounding, factual_grounding_metrics = structured_reconstruction_loss(
            model.tokenizer,
            factual_decoded,
            nxt,
            nglob,
            cell_weights=_changed_cell_weights(
                state, nxt, changed_token_boost
            ),
        )
        grounding_metrics.update(
            _grounding_metrics(factual_grounding_metrics, "factual")
        )

    counterfactual = factual.new_zeros(())
    effect = factual.new_zeros(())
    effect_cosine = factual.new_zeros(())
    effect_norm_ratio = factual.new_zeros(())
    counterfactual_grounding = factual.new_zeros(())
    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    if cf_valid.any():
        idx = torch.nonzero(cf_valid, as_tuple=False).flatten()
        raw_cf_state = batch["counterfactual_next_state"].reshape(
            -1, *batch["counterfactual_next_state"].shape[-2:]
        )
        raw_cf_glob = batch["counterfactual_next_globals"].reshape(
            -1, batch["counterfactual_next_globals"].shape[-1]
        )
        # Encode at the factual batch shape before subselecting. This preserves
        # exact zero effects for semantically identical cloned arrivals.
        cf_state_all = torch.where(cf_valid[:, None, None], raw_cf_state, nxt)
        cf_glob_all = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
        with torch.no_grad():
            target_zcf = _normalized(
                model.target_tokenizer.encode(cf_state_all, cf_glob_all)
            )[idx]
        cf_action = batch["counterfactual_action"].reshape(
            -1, *batch["counterfactual_action"].shape[-2:]
        )[idx]
        cf_opponent = batch["counterfactual_opponent_action"].reshape(
            -1, *batch["counterfactual_opponent_action"].shape[-2:]
        )[idx]
        cf_events, cf_event_valid, cf_overflow = model.predictor.events(
            state[idx], cf_action, cf_opponent
        )
        overflow = torch.cat((overflow.reshape(-1), cf_overflow.reshape(-1)))
        cf_delta = model.predictor(z0[idx], cf_events, cf_event_valid)
        if model.jepa_cfg.raw_prediction:
            cf_raw = raw_z0[idx] + cf_delta
            cf_pred = _normalized(cf_raw)
        else:
            cf_raw = None
            cf_pred = z0[idx] + cf_delta
        effect_target = target_zcf - target_z1[idx]
        cf_weights, _, _ = _token_weights(
            model.tokenizer,
            (nxt[idx], cf_state_all[idx]),
            target_zcf - target_z0[idx],
            changed_token_boost,
            change_threshold,
            padding_token_weight,
        )
        effect_weights, _, effect_changed = _token_weights(
            model.tokenizer,
            (nxt[idx], cf_state_all[idx]),
            effect_target,
            changed_token_boost,
            change_threshold,
            padding_token_weight,
        )
        counterfactual = _weighted_mse(cf_pred, target_zcf, cf_weights)
        predicted_effect = cf_pred - factual_pred[idx]
        effect = _weighted_mse(predicted_effect, effect_target, effect_weights)
        has_effect = effect_changed.any(1)
        if has_effect.any():
            pred_flat = predicted_effect[has_effect].float().flatten(1)
            target_flat = effect_target[has_effect].float().flatten(1)
            effect_cosine = F.cosine_similarity(pred_flat, target_flat, dim=1).mean()
            effect_norm_ratio = pred_flat.norm(dim=1).sum() / target_flat.norm(
                dim=1
            ).sum().clamp_min(1.0e-8)
        if counterfactual_grounding_coef:
            cf_decoded = model.tokenizer.decode(cf_raw)
            (
                counterfactual_grounding,
                counterfactual_grounding_metrics,
            ) = structured_reconstruction_loss(
                model.tokenizer,
                cf_decoded,
                cf_state_all[idx],
                cf_glob_all[idx],
                cell_weights=_changed_cell_weights(
                    state[idx], cf_state_all[idx], changed_token_boost
                ),
            )
            grounding_metrics.update(
                _grounding_metrics(counterfactual_grounding_metrics, "counterfactual")
            )

    total = (
        float(reconstruction_coef) * reconstruction
        + float(factual_coef) * factual
        + float(counterfactual_coef) * counterfactual
        + float(effect_coef) * effect
        + float(factual_grounding_coef) * factual_grounding
        + float(counterfactual_grounding_coef) * counterfactual_grounding
    )
    metrics = dict(recon_metrics)
    metrics.update(
        {
            "tok/reconstruction": reconstruction.detach(),
            "tok/jepa_factual": factual.detach(),
            "tok/jepa_counterfactual": counterfactual.detach(),
            "tok/jepa_effect": effect.detach(),
            "tok/jepa_factual_grounding": factual_grounding.detach(),
            "tok/jepa_counterfactual_grounding": counterfactual_grounding.detach(),
            "tok/jepa_effect_cosine": effect_cosine.detach(),
            "tok/jepa_effect_norm_ratio_aggregate": effect_norm_ratio.detach(),
            "tok/jepa_changed_fraction": (
                changed.sum().float() / valid_tokens.sum().clamp_min(1)
            ).detach(),
            "tok/jepa_counterfactual_fraction": cf_valid.float().mean().detach(),
            "tok/jepa_action_overflow": overflow.float().mean().detach(),
            "tok/jepa_target_drift": F.mse_loss(z0.detach().float(), target_z0.float()),
            "tok/total": total.detach(),
        }
    )
    metrics.update(grounding_metrics)
    return total, metrics
