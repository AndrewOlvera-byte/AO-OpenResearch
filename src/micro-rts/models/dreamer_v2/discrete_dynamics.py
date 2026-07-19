"""No-flow causal language model over hard MicroRTS state codes."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_tokenizer import FactorizedActionEventEncoder
from .config import DiscreteDynamicsConfig, DiscreteTokenizerConfig
from .discrete_action_tokenizer import DiscreteActionCodeRouter, _depth_logits
from .discrete_tokenizer import DiscreteStructuredTokenizer
from .schema import dense_actions_to_events


class DiscreteCausalTransformer(nn.Module):
    """Prefix-conditioned autoregressive next-code transformer.

    Sequence layout is ``[state codes, action events, BOS_NEXT, next codes]``.
    Loss is applied only to next-code positions.  There are no flow signals,
    target noise, latent normalization, shortcut steps, or Euler integration.
    """

    def __init__(
        self,
        tokenizer: DiscreteStructuredTokenizer,
        grid_hw=(16, 16),
        cfg: DiscreteDynamicsConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or DiscreteDynamicsConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        self.n_code_tokens = tokenizer.n_code_tokens
        self.n_semantic_tokens = tokenizer.n_tokens
        self.n_spatial = tokenizer.n_spatial
        self.n_global = tokenizer.n_global
        self.codebook_depth = tokenizer.codebook_depth
        self.codebook_size = tokenizer.codebook_size
        self.spatial_downsample = tokenizer.cfg.spatial_downsample
        d = int(self.cfg.d_model)

        self.code_embeddings = nn.Parameter(
            torch.empty(self.codebook_depth, self.codebook_size, d)
        )
        nn.init.normal_(self.code_embeddings, std=1.0 / math.sqrt(d))
        self.depth_embed = nn.Embedding(self.codebook_depth, d)
        spatial_h = self.grid_hw[0] // self.spatial_downsample
        spatial_w = self.grid_hw[1] // self.spatial_downsample
        self.x_embed = nn.Embedding(spatial_w + 1, d)
        self.y_embed = nn.Embedding(spatial_h + 1, d)
        self.slot_type = nn.Embedding(2, d)
        self.semantic_embed = nn.Embedding(self.n_semantic_tokens, d)
        self.type_embed = nn.Embedding(3, d)  # state, action, target
        self.bos_next = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.bos_next, std=0.02)

        semantic = torch.arange(self.n_semantic_tokens)
        spatial = semantic < self.n_spatial
        x = torch.where(spatial, semantic % spatial_w, spatial_w)
        y = torch.where(spatial, semantic // spatial_w, spatial_h)
        slot = (~spatial).long()
        self.register_buffer("semantic_x", x, persistent=False)
        self.register_buffer("semantic_y", y, persistent=False)
        self.register_buffer("semantic_type", slot, persistent=False)

        self.action_encoder = FactorizedActionEventEncoder(
            d,
            self.grid_hw,
            self.cfg.max_unit_types,
            self.cfg.action_field_dim,
        )
        self.action_position = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        nn.init.normal_(self.action_position, std=0.02)
        self.action_router = (
            DiscreteActionCodeRouter(
                tokenizer.d_latent,
                d,
                self.codebook_depth,
                self.codebook_size,
                self.cfg.n_heads,
                self.cfg.mlp_ratio,
                self.cfg.dropout,
            )
            if self.cfg.pretrained_action_router
            else None
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
        self.transformer = nn.TransformerEncoder(layer, self.cfg.depth, nn.LayerNorm(d))
        self.correction_heads = nn.ModuleList(
            nn.Linear(d, self.codebook_size) for _ in range(self.codebook_depth)
        )
        if self.cfg.zero_init_correction:
            for head in self.correction_heads:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
        max_len = 2 * self.n_code_tokens + self.cfg.max_action_events + 1
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def _position(self, batch: int, device) -> torch.Tensor:
        semantic = (
            self.x_embed(self.semantic_x)
            + self.y_embed(self.semantic_y)
            + self.slot_type(self.semantic_type)
            + self.semantic_embed.weight
        )
        semantic = semantic[:, None, :] + self.depth_embed.weight[None, :, :]
        return semantic.reshape(1, self.n_code_tokens, -1).expand(batch, -1, -1)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        flat = codes.reshape(
            codes.shape[0], self.n_semantic_tokens, self.codebook_depth
        )
        depth = torch.arange(self.codebook_depth, device=codes.device)
        embedded = self.code_embeddings[depth[None, None, :], flat]
        return embedded.reshape(codes.shape[0], self.n_code_tokens, -1)

    def encode_actions(self, events):
        return self.action_encoder(events) + self.action_position[: events.shape[1]]

    def forward(
        self,
        current_codes: torch.Tensor,
        target_codes: torch.Tensor,
        events: torch.Tensor,
        event_valid: torch.Tensor,
        *,
        router_state_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if current_codes.shape != target_codes.shape:
            raise ValueError(
                "current and target code tensors must have identical shape"
            )
        batch = current_codes.shape[0]
        current = current_codes.reshape(batch, -1)
        target = target_codes.reshape(batch, -1)
        if current.shape[1] != self.n_code_tokens:
            raise ValueError("state code count mismatch")
        position = self._position(batch, current.device)
        state_tokens = self.embed_codes(current) + position + self.type_embed.weight[0]
        action_tokens = self.encode_actions(events)
        action_sequence = action_tokens + self.type_embed.weight[1]

        shifted = target.new_zeros(target.shape)
        shifted[:, 1:] = target[:, :-1]
        target_inputs = self.embed_codes(shifted)
        target_inputs[:, 0] = self.bos_next
        target_inputs = target_inputs + position + self.type_embed.weight[2]
        sequence = torch.cat((state_tokens, action_sequence, target_inputs), dim=1)
        length = sequence.shape[1]
        padding = torch.zeros(batch, length, dtype=torch.bool, device=sequence.device)
        lo = self.n_code_tokens
        padding[:, lo : lo + events.shape[1]] = ~event_valid
        hidden = self.transformer(
            sequence,
            mask=self.causal_mask[:length, :length],
            src_key_padding_mask=padding,
        )
        target_hidden = hidden[:, -self.n_code_tokens :]
        correction = _depth_logits(self.correction_heads, target_hidden)
        if self.action_router is None:
            prior = torch.zeros_like(correction)
        else:
            if router_state_tokens is None:
                raise ValueError(
                    "pretrained_action_router requires tokenizer code embeddings"
                )
            prior = self.action_router(router_state_tokens, action_tokens, event_valid)
        return {
            "logits": prior + correction,
            "prior_logits": prior,
            "correction_logits": correction,
            "target_hidden": target_hidden,
        }

    @torch.no_grad()
    def generate(
        self,
        current_codes,
        events,
        event_valid,
        *,
        router_state_tokens=None,
        temperature: float = 0.0,
    ):
        batch = current_codes.shape[0]
        generated = torch.zeros(
            batch, self.n_code_tokens, dtype=torch.long, device=current_codes.device
        )
        shaped = generated.reshape(batch, self.n_semantic_tokens, self.codebook_depth)
        for index in range(self.n_code_tokens):
            output = self(
                current_codes,
                shaped,
                events,
                event_valid,
                router_state_tokens=router_state_tokens,
            )["logits"][:, index]
            if temperature > 0:
                probs = (output / float(temperature)).softmax(-1)
                generated[:, index] = torch.multinomial(probs, 1).squeeze(-1)
            else:
                generated[:, index] = output.argmax(-1)
        return generated.reshape(batch, self.n_semantic_tokens, self.codebook_depth)


class DiscreteStructuredWorldModel(nn.Module):
    def __init__(
        self,
        grid_hw=(16, 16),
        tokenizer_cfg: DiscreteTokenizerConfig | None = None,
        dynamics_cfg: DiscreteDynamicsConfig | None = None,
    ):
        super().__init__()
        self.tokenizer = DiscreteStructuredTokenizer(grid_hw, tokenizer_cfg)
        self.dynamics = DiscreteCausalTransformer(self.tokenizer, grid_hw, dynamics_cfg)
        self.grid_hw = tuple(map(int, grid_hw))

    def action_events(self, state, action, opponent_action):
        return dense_actions_to_events(
            state,
            action,
            opponent_action,
            self.grid_hw,
            self.dynamics.cfg.max_action_events,
        )

    def router_tokens(self, codes):
        return self.tokenizer.embed_code_tokens(codes)

    @torch.no_grad()
    def generate_next(self, state, globals_, action, opponent_action, temperature=0.0):
        codes = self.tokenizer.encode_codes(state, globals_)
        events, valid, _ = self.action_events(state, action, opponent_action)
        nxt = self.dynamics.generate(
            codes,
            events,
            valid,
            router_state_tokens=self.router_tokens(codes),
            temperature=temperature,
        )
        return self.tokenizer.discretize(self.tokenizer.decode_codes(nxt)), nxt


def load_discrete_action_jepa(model: DiscreteStructuredWorldModel, payload: dict):
    """Transfer the exact event encoder, positions, and categorical router."""
    state = payload.get("model", payload)
    prefixes = {
        "action_encoder.": model.dynamics.action_encoder,
        "router.": model.dynamics.action_router,
    }
    for prefix, module in prefixes.items():
        if module is None:
            raise ValueError("checkpoint contains a router but dynamics disabled it")
        subset = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not subset:
            raise ValueError(f"action-JEPA checkpoint lacks {prefix} weights")
        module.load_state_dict(subset, strict=True)
    model.dynamics.action_position.data.copy_(state["action_position"])
    return payload


@torch.no_grad()
def discrete_prior_geometry(
    model: DiscreteStructuredWorldModel, batch: dict
) -> dict[str, torch.Tensor]:
    """Audit the transferred categorical action prior before dynamics training.

    The paired preference is the discrete analogue of effect-vector cosine: on
    every code changed by a cloned intervention, the factual action should favor
    the factual code and the counterfactual action should favor its own code.
    Reporting this directly prevents broad unchanged-code accuracy from hiding
    an action-insensitive initialization.
    """
    router = model.dynamics.action_router
    if router is None:
        raise ValueError("categorical prior geometry requires an action router")

    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opponent = batch["opponent_action"].reshape(
        -1, *batch["opponent_action"].shape[-2:]
    )
    c0 = model.tokenizer.encode_codes(state, glob)
    c1 = model.tokenizer.encode_codes(nxt, nglob)
    flat0, flat1 = c0.flatten(1), c1.flatten(1)
    events, valid, overflow = model.action_events(state, action, opponent)
    factual_logits = router(
        model.router_tokens(c0), model.dynamics.encode_actions(events), valid
    )
    factual_pred = factual_logits.argmax(-1)
    changed = flat1 != flat0
    changed_count = changed.sum()
    zero = factual_logits.new_zeros(())

    metrics = {
        "geometry/code_acc": (factual_pred == flat1).float().mean(),
        "geometry/copy_code_acc": (flat0 == flat1).float().mean(),
        "geometry/changed_code_acc": (
            (factual_pred[changed] == flat1[changed]).float().mean()
            if changed.any()
            else zero
        ),
        "geometry/changed_codes": changed_count,
        "geometry/action_overflow": overflow.float().mean(),
        "geometry/paired_rows": zero,
        "geometry/paired_effect_codes": zero,
        "geometry/factual_preference": zero,
        "geometry/counterfactual_preference": zero,
        "geometry/bidirectional_preference": zero,
        "geometry/preference_margin": zero,
        "geometry/branch_disagreement": zero,
    }

    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    metrics["geometry/paired_rows"] = cf_valid.sum()
    if not cf_valid.any():
        return metrics

    idx = torch.nonzero(cf_valid, as_tuple=False).flatten()
    raw_cf_state = batch["counterfactual_next_state"].reshape(
        -1, *batch["counterfactual_next_state"].shape[-2:]
    )
    raw_cf_glob = batch["counterfactual_next_globals"].reshape(
        -1, batch["counterfactual_next_globals"].shape[-1]
    )
    aligned_state = torch.where(cf_valid[:, None, None], raw_cf_state, nxt)
    aligned_glob = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
    cf_codes = model.tokenizer.encode_codes(aligned_state, aligned_glob)[idx].flatten(1)
    cf_action = batch["counterfactual_action"].reshape(
        -1, *batch["counterfactual_action"].shape[-2:]
    )[idx]
    cf_opponent = batch["counterfactual_opponent_action"].reshape(
        -1, *batch["counterfactual_opponent_action"].shape[-2:]
    )[idx]
    cf_events, cf_event_valid, cf_overflow = model.action_events(
        state[idx], cf_action, cf_opponent
    )
    metrics["geometry/action_overflow"] = torch.maximum(
        metrics["geometry/action_overflow"], cf_overflow.float().mean()
    )
    cf_logits = router(
        model.router_tokens(c0[idx]),
        model.dynamics.encode_actions(cf_events),
        cf_event_valid,
    )
    factual_paired = factual_logits[idx]
    factual_target = flat1[idx]
    effect = factual_target != cf_codes
    metrics["geometry/paired_effect_codes"] = effect.sum()
    if not effect.any():
        return metrics

    factual_logp = factual_paired.log_softmax(-1)
    cf_logp = cf_logits.log_softmax(-1)
    factual_index = factual_target.unsqueeze(-1)
    cf_index = cf_codes.unsqueeze(-1)
    factual_margin = (
        factual_logp.gather(-1, factual_index).squeeze(-1)
        - factual_logp.gather(-1, cf_index).squeeze(-1)
    )[effect]
    cf_margin = (
        cf_logp.gather(-1, cf_index).squeeze(-1)
        - cf_logp.gather(-1, factual_index).squeeze(-1)
    )[effect]
    metrics.update(
        {
            "geometry/factual_preference": (factual_margin > 0).float().mean(),
            "geometry/counterfactual_preference": (cf_margin > 0).float().mean(),
            "geometry/bidirectional_preference": (
                (factual_margin > 0) & (cf_margin > 0)
            )
            .float()
            .mean(),
            "geometry/preference_margin": (factual_margin + cf_margin).mean() / 2,
            "geometry/branch_disagreement": (
                factual_paired.argmax(-1)[effect] != cf_logits.argmax(-1)[effect]
            )
            .float()
            .mean(),
        }
    )
    return metrics


def _balanced_ce(logits, target, current):
    per = F.cross_entropy(
        logits.flatten(0, -2), target.flatten(), reduction="none"
    ).reshape_as(target)
    changed = target != current
    unchanged = ~changed
    lc = per[changed].mean() if changed.any() else per.new_zeros(())
    lu = per[unchanged].mean() if unchanged.any() else per.new_zeros(())
    return lc + lu, lc, lu, changed


def _effect_margin(factual_logits, cf_logits, factual_target, cf_target, margin):
    effect = factual_target != cf_target
    if not effect.any():
        return factual_logits.new_zeros(()), effect
    lf, lc = factual_logits.log_softmax(-1), cf_logits.log_softmax(-1)
    yf, yc = factual_target.unsqueeze(-1), cf_target.unsqueeze(-1)
    factual_pref = lf.gather(-1, yf).squeeze(-1) - lf.gather(-1, yc).squeeze(-1)
    cf_pref = lc.gather(-1, yc).squeeze(-1) - lc.gather(-1, yf).squeeze(-1)
    loss = F.relu(float(margin) - factual_pref[effect]).mean()
    loss = loss + F.relu(float(margin) - cf_pref[effect]).mean()
    return loss, effect


def discrete_causal_paired_loss(
    model: DiscreteStructuredWorldModel,
    batch: dict,
    *,
    factual_coef: float = 1.0,
    counterfactual_coef: float = 1.0,
    effect_margin_coef: float = 2.0,
    effect_margin: float = 1.0,
):
    """Teacher-forced next-code likelihood with explicit intervention margin."""
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opponent = batch["opponent_action"].reshape(
        -1, *batch["opponent_action"].shape[-2:]
    )
    with torch.no_grad():
        c0 = model.tokenizer.encode_codes(state, glob)
        c1 = model.tokenizer.encode_codes(nxt, nglob)
        router_tokens = model.router_tokens(c0)
    flat0, flat1 = c0.flatten(1), c1.flatten(1)
    events, valid, overflow = model.action_events(state, action, opponent)
    factual_out = model.dynamics(
        c0, c1, events, valid, router_state_tokens=router_tokens
    )
    factual, changed_loss, unchanged_loss, changed = _balanced_ce(
        factual_out["logits"], flat1, flat0
    )

    counterfactual = factual.new_zeros(())
    effect_loss = factual.new_zeros(())
    effect_fraction = factual.new_zeros(())
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
            ccf_all = model.tokenizer.encode_codes(aligned_state, aligned_glob)
        ccf = ccf_all[idx]
        cf_action = batch["counterfactual_action"].reshape(
            -1, *batch["counterfactual_action"].shape[-2:]
        )[idx]
        cf_opp = batch["counterfactual_opponent_action"].reshape(
            -1, *batch["counterfactual_opponent_action"].shape[-2:]
        )[idx]
        cf_events, cf_valid_events, _ = model.action_events(
            state[idx], cf_action, cf_opp
        )
        cf_out = model.dynamics(
            c0[idx],
            ccf,
            cf_events,
            cf_valid_events,
            router_state_tokens=router_tokens[idx],
        )
        flat_cf = ccf.flatten(1)
        counterfactual = _balanced_ce(cf_out["logits"], flat_cf, flat0[idx])[0]
        # Compare branches under a common factual target prefix. Otherwise the
        # paired margin can be satisfied by teacher-forced branch history after
        # the first divergence instead of by the alternative action itself.
        cf_common_prefix = model.dynamics(
            c0[idx],
            c1[idx],
            cf_events,
            cf_valid_events,
            router_state_tokens=router_tokens[idx],
        )
        effect_loss, effect = _effect_margin(
            factual_out["logits"][idx],
            cf_common_prefix["logits"],
            flat1[idx],
            flat_cf,
            effect_margin,
        )
        effect_fraction = effect.float().mean()

    total = (
        float(factual_coef) * factual
        + float(counterfactual_coef) * counterfactual
        + float(effect_margin_coef) * effect_loss
    )
    with torch.no_grad():
        pred_flat = factual_out["logits"].argmax(-1)
        pred_codes = pred_flat.reshape_as(c1)
        pred_state, pred_globals = model.tokenizer.discretize(
            model.tokenizer.decode_codes(pred_codes)
        )
        current_state = state.clone()
        current_state[..., 2] = -1
        true_state = nxt.clone()
        true_state[..., 2] = -1
        exact_cell = (pred_state == true_state).all(-1)
        true_changed = (current_state != true_state).any(-1)
        pred_changed = (current_state != pred_state).any(-1)
        tp = (true_changed & pred_changed).sum().float()
        fp = (~true_changed & pred_changed).sum().float()
        fn = (true_changed & ~pred_changed).sum().float()
        changed_f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        changed_acc = (
            (pred_flat[changed] == flat1[changed]).float().mean()
            if changed.any()
            else total.new_ones(())
        )
    return total, {
        "dynamics/total": total.detach(),
        "dynamics/factual": factual.detach(),
        "dynamics/factual_changed": changed_loss.detach(),
        "dynamics/factual_unchanged": unchanged_loss.detach(),
        "dynamics/counterfactual": counterfactual.detach(),
        "dynamics/effect_margin": effect_loss.detach(),
        "dynamics/code_acc": (pred_flat == flat1).float().mean(),
        "dynamics/changed_code_acc": changed_acc,
        "dynamics/exact_cell_teacher_forced": exact_cell.float().mean(),
        "dynamics/exact_frame_teacher_forced": exact_cell.all(-1).float().mean(),
        "dynamics/exact_globals_teacher_forced": (pred_globals == nglob)
        .all(-1)
        .float()
        .mean(),
        "dynamics/changed_cell_f1_teacher_forced": changed_f1,
        "dynamics/effect_code_fraction": effect_fraction.detach(),
        "dynamics/counterfactual_fraction": cf_valid.float().mean(),
        "dynamics/action_overflow": overflow.float().mean(),
    }
