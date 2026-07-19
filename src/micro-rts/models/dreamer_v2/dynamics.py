"""Inspectable causal flow-matching dynamics for structured MicroRTS tokens."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StructuredDynamicsConfig, StructuredTokenizerConfig
from .action_tokenizer import FactorizedActionEventEncoder
from .schema import dense_actions_to_events
from .tokenizer import StructuredTokenizer, structured_reconstruction_loss


class ActionEventEncoder(nn.Module):
    """Sparse source->target joint-action records -> transformer tokens."""

    def __init__(self, d_model: int, grid_hw=(16, 16), max_unit_types=16):
        super().__init__()
        h, w = grid_hw
        self.h, self.w = int(h), int(w)
        self.role = nn.Embedding(3, d_model)
        self.atype = nn.Embedding(6, d_model)
        self.direction = nn.Embedding(6, d_model)  # -1..4 shifted
        self.produced = nn.Embedding(max_unit_types + 1, d_model)
        self.src_x = nn.Embedding(w, d_model)
        self.src_y = nn.Embedding(h, d_model)
        self.dst_x = nn.Embedding(w + 2, d_model)  # clamp + sentinel bucket
        self.dst_y = nn.Embedding(h + 2, d_model)
        self.numeric = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        role, _uid, sx, sy, typ, tx, ty, direction, produced, attack = events.unbind(-1)
        x = self.role(role.clamp(0, 2)) + self.atype(typ.clamp(0, 5))
        x = x + self.direction((direction + 1).clamp(0, 5))
        x = x + self.produced((produced + 1).clamp(0, self.produced.num_embeddings - 1))
        x = (
            x
            + self.src_x(sx.clamp(0, self.w - 1))
            + self.src_y(sy.clamp(0, self.h - 1))
        )
        x = x + self.dst_x((tx + 1).clamp(0, self.w + 1))
        x = x + self.dst_y((ty + 1).clamp(0, self.h + 1))
        # Raw unit IDs remain in event records for debugging only. They are JVM
        # allocation counters; role + source coordinate is the Markov binding.
        nums = attack.float().unsqueeze(-1) / 49.0
        return self.norm(x + self.numeric(nums))


class CausalStructuredWorldModel(nn.Module):
    """One Markov transition packed as a conventional causal token sequence.

    Sequence: [departure state, sparse joint actions, signal, step, noised arrival].
    Arrival outputs can attend to the complete departure and action prefix but
    never to clean future tokens.
    """

    def __init__(
        self,
        n_state_tokens: int,
        d_latent: int,
        grid_hw=(16, 16),
        max_unit_types=16,
        cfg: StructuredDynamicsConfig | None = None,
        *,
        n_spatial_tokens: int | None = None,
        n_entity_tokens: int | None = None,
        spatial_downsample: int = 1,
    ):
        super().__init__()
        self.cfg = cfg or StructuredDynamicsConfig()
        self.grid_hw = tuple(map(int, grid_hw))
        self.n_state_tokens = int(n_state_tokens)
        self.d_latent = int(d_latent)
        self.n_spatial_tokens = int(n_spatial_tokens or 0)
        self.n_entity_tokens = int(n_entity_tokens or 0)
        self.spatial_downsample = int(spatial_downsample)
        self.k_max = int(self.cfg.k_max)
        if self.k_max < 2 or self.k_max & (self.k_max - 1):
            raise ValueError("k_max must be a power of two")
        if self.cfg.initial_noise not in ("gaussian", "zero"):
            raise ValueError(
                "initial_noise must be 'gaussian' or 'zero', got "
                f"{self.cfg.initial_noise!r}"
            )
        self.emax = int(math.log2(self.k_max))
        d = self.cfg.d_model
        self.state_in = nn.Linear(d_latent, d)
        self.target_in = nn.Linear(d_latent, d)
        if self.cfg.action_encoder_type == "legacy":
            self.action_encoder = ActionEventEncoder(d, grid_hw, max_unit_types)
        elif self.cfg.action_encoder_type == "factorized":
            self.action_encoder = FactorizedActionEventEncoder(
                d,
                grid_hw,
                max_unit_types,
                field_dim=self.cfg.action_field_dim,
            )
        else:
            raise ValueError(
                "action_encoder_type must be 'legacy' or 'factorized', got "
                f"{self.cfg.action_encoder_type!r}"
            )
        if self.cfg.pretrained_action_router and not self.cfg.residual_prediction:
            raise ValueError(
                "pretrained_action_router requires residual_prediction=true"
            )
        if self.cfg.explicit_spatial_action_routing and not self.n_spatial_tokens:
            raise ValueError(
                "explicit_spatial_action_routing requires spatial token metadata"
            )
        self.signal_embed = nn.Embedding(self.k_max, d)
        self.step_embed = nn.Embedding(self.emax + 1, d)
        self.type_embed = nn.Embedding(5, d)  # state, action, signal, step, target
        self.state_position = nn.Parameter(torch.zeros(n_state_tokens, d))
        self.action_position = nn.Parameter(torch.zeros(self.cfg.max_action_events, d))
        nn.init.normal_(self.state_position, std=0.02)
        nn.init.normal_(self.action_position, std=0.02)
        if self.cfg.pretrained_action_router:
            # These modules intentionally mirror ActionTokenizerPretrainer's
            # forward path so its learned action-to-state bridge transfers
            # exactly instead of being discarded after SSL pretraining.
            self.action_router_state = nn.Linear(d_latent, d)
            self.action_router_attn = nn.MultiheadAttention(
                d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
            )
            self.action_router_norm = nn.LayerNorm(d)
            self.action_router_head = nn.Sequential(
                nn.Linear(d, int(self.cfg.mlp_ratio * d)),
                nn.GELU(),
                nn.Linear(int(self.cfg.mlp_ratio * d), d_latent),
            )
        else:
            self.action_router_state = None
            self.action_router_attn = None
            self.action_router_norm = None
            self.action_router_head = None
        if self.cfg.explicit_spatial_action_routing:
            self.source_spatial_route = nn.Linear(d, d, bias=False)
            self.target_spatial_route = nn.Linear(d, d, bias=False)
            # Preserve the pretrained router's geometry at initialization; the
            # transition optimizer learns only the explicit routing correction.
            nn.init.zeros_(self.source_spatial_route.weight)
            nn.init.zeros_(self.target_spatial_route.weight)
        else:
            self.source_spatial_route = None
            self.target_spatial_route = None
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
        self.flow_x_head = nn.Linear(d, d_latent)
        self.shortcut_skip_head = nn.Linear(d, d_latent)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)
        nn.init.zeros_(self.shortcut_skip_head.weight)
        nn.init.zeros_(self.shortcut_skip_head.bias)
        self.register_buffer("latent_mean", torch.zeros(1, 1, d_latent))
        self.register_buffer("latent_std", torch.ones(1, 1, d_latent))
        max_len = 2 * n_state_tokens + self.cfg.max_action_events + 2
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1),
            persistent=False,
        )

    def set_latent_stats(self, mean: torch.Tensor, std: torch.Tensor):
        self.latent_mean.copy_(mean.reshape(1, 1, -1))
        self.latent_std.copy_(std.reshape(1, 1, -1).clamp_min(1e-4))

    def normalize(self, z):
        return (z - self.latent_mean) / self.latent_std

    def denormalize(self, z):
        return z * self.latent_std + self.latent_mean

    def initial_noise_like(self, reference: torch.Tensor) -> torch.Tensor:
        if self.cfg.initial_noise == "zero":
            return torch.zeros_like(reference)
        return torch.randn_like(reference)

    def encode_action_tokens(self, events: torch.Tensor) -> torch.Tensor:
        return self.action_encoder(events) + self.action_position[: events.shape[1]]

    def action_prior_delta(
        self,
        departure_z: torch.Tensor,
        action_tokens: torch.Tensor,
        event_valid: torch.Tensor,
    ) -> torch.Tensor:
        if self.action_router_state is None:
            return torch.zeros_like(departure_z)
        query = self.action_router_state(departure_z)
        routed = self.action_router_attn(
            query,
            action_tokens,
            action_tokens,
            key_padding_mask=~event_valid,
            need_weights=False,
        )[0]
        return self.action_router_head(self.action_router_norm(query + routed))

    def spatial_action_route(
        self,
        events: torch.Tensor,
        action_tokens: torch.Tensor,
        event_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Scatter event features into source/destination spatial token slots."""
        if self.source_spatial_route is None:
            return action_tokens.new_zeros(
                action_tokens.shape[0], self.n_state_tokens, action_tokens.shape[-1]
            )
        h, w = map(int, self.grid_hw)
        ds = self.spatial_downsample
        spatial_w = w // ds
        sx, sy = events[..., 2], events[..., 3]
        tx, ty = events[..., 5], events[..., 6]
        src = (sy.clamp(0, h - 1) // ds) * spatial_w + sx.clamp(0, w - 1) // ds
        dst = (ty.clamp(0, h - 1) // ds) * spatial_w + tx.clamp(0, w - 1) // ds
        src_valid = event_valid & (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        dst_valid = event_valid & (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h)

        source_values = self.source_spatial_route(action_tokens)
        target_values = self.target_spatial_route(action_tokens)
        # Under AMP, linear projections are BF16 even though the embedding that
        # produced ``action_tokens`` remains FP32. Allocate the accumulator from
        # the projected values so scatter_add_ receives matching dtypes.
        route = source_values.new_zeros(
            action_tokens.shape[0], self.n_state_tokens, action_tokens.shape[-1]
        )

        def scatter(index, values, valid):
            weighted = values * valid[..., None].to(values.dtype)
            route.scatter_add_(1, index[..., None].expand_as(weighted), weighted)

        scatter(src, source_values, src_valid)
        scatter(dst, target_values, dst_valid)
        return route

    def forward(
        self,
        departure_z: torch.Tensor,
        noised_arrival_z: torch.Tensor,
        events: torch.Tensor,
        event_valid: torch.Tensor,
        signal_idx: torch.Tensor,
        step_idx: torch.Tensor,
        state_token_valid: torch.Tensor | None = None,
    ) -> dict:
        b, ns, _ = departure_z.shape
        if ns != self.n_state_tokens or noised_arrival_z.shape[1] != ns:
            raise ValueError("state token count mismatch")
        st = (
            self.state_in(departure_z) + self.type_embed.weight[0] + self.state_position
        )
        action_tokens = self.encode_action_tokens(events)
        at = action_tokens + self.type_embed.weight[1]
        sig = self.signal_embed(signal_idx.long())[:, None] + self.type_embed.weight[2]
        step = self.step_embed(step_idx.long())[:, None] + self.type_embed.weight[3]
        tgt = (
            self.target_in(noised_arrival_z)
            + self.type_embed.weight[4]
            + self.state_position
        )
        tgt = tgt + self.spatial_action_route(events, action_tokens, event_valid)
        tokens = torch.cat([st, at, sig, step, tgt], dim=1)
        length = tokens.shape[1]
        causal = self.causal_mask[:length, :length]
        pad = torch.zeros(b, length, dtype=torch.bool, device=tokens.device)
        pad[:, ns : ns + events.shape[1]] = ~event_valid
        if self.cfg.mask_empty_entity_tokens:
            if state_token_valid is None or state_token_valid.shape != (b, ns):
                raise ValueError(
                    "mask_empty_entity_tokens requires state_token_valid with "
                    f"shape {(b, ns)}"
                )
            # Mask only departure keys. Target queries must remain live so newly
            # spawned or removed entity slots can still be predicted.
            pad[:, :ns] = ~state_token_valid
        h = self.transformer(tokens, mask=causal, src_key_padding_mask=pad)
        ht = h[:, -ns:]
        correction = self.flow_x_head(ht)
        prior_delta = self.action_prior_delta(
            departure_z, action_tokens, event_valid
        )
        base = correction
        if self.cfg.residual_prediction:
            base = departure_z + prior_delta + correction
        skip = self.shortcut_skip_head(ht)
        return {
            "x": base + skip,
            "base": base,
            "skip": skip,
            "target_h": ht,
            "prior_delta": prior_delta,
            "correction": correction,
        }

    @torch.no_grad()
    def sample_next(
        self,
        departure_z: torch.Tensor,
        events: torch.Tensor,
        event_valid: torch.Tensor,
        steps: int = 4,
        initial_noise: torch.Tensor | None = None,
        state_token_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        k = int(steps)
        if k < 1 or k > self.k_max or k & (k - 1):
            raise ValueError("steps must be a power of two <= k_max")
        departure_z = self.normalize(departure_z)
        z = (
            self.initial_noise_like(departure_z)
            if initial_noise is None
            else initial_noise.clone()
        )
        e = int(math.log2(k))
        for i in range(k):
            tau = i / k
            sig = torch.full(
                (z.shape[0],), (i * self.k_max) // k, dtype=torch.long, device=z.device
            )
            step = torch.full_like(sig, e)
            x = self(
                departure_z,
                z,
                events,
                event_valid,
                sig,
                step,
                state_token_valid=state_token_valid,
            )["x"]
            z = z + (1.0 / k) * (x - z) / max(1.0 - tau, 1e-4)
        return self.denormalize(z)


class StructuredWorldModelV2(nn.Module):
    def __init__(
        self,
        grid_hw=(16, 16),
        tokenizer_cfg: StructuredTokenizerConfig | None = None,
        dynamics_cfg: StructuredDynamicsConfig | None = None,
    ):
        super().__init__()
        self.tokenizer = StructuredTokenizer(grid_hw, tokenizer_cfg)
        self.dynamics = CausalStructuredWorldModel(
            self.tokenizer.n_tokens,
            self.tokenizer.d_latent,
            grid_hw,
            self.tokenizer.cfg.max_unit_types,
            dynamics_cfg,
            n_spatial_tokens=self.tokenizer.n_spatial,
            n_entity_tokens=self.tokenizer.n_entity,
            spatial_downsample=self.tokenizer.cfg.downsample,
        )
        self.grid_hw = tuple(grid_hw)

    def action_events(self, state, action, opponent_action):
        return dense_actions_to_events(
            state,
            action,
            opponent_action,
            self.grid_hw,
            self.dynamics.cfg.max_action_events,
        )

    def state_token_valid(self, state: torch.Tensor) -> torch.Tensor:
        """Validity mask for spatial, packed-entity, and global state tokens."""
        lead = state.shape[:-2]
        flat = state.reshape(-1, *state.shape[-2:])
        valid = torch.ones(
            flat.shape[0],
            self.tokenizer.n_tokens,
            dtype=torch.bool,
            device=state.device,
        )
        count = flat[..., 1].bool().sum(1).clamp_max(self.tokenizer.n_entity)
        entity_index = torch.arange(self.tokenizer.n_entity, device=state.device)[None]
        start = self.tokenizer.n_spatial
        valid[:, start : start + self.tokenizer.n_entity] = entity_index < count[:, None]
        return valid.reshape(*lead, self.tokenizer.n_tokens)


@torch.no_grad()
def structured_action_gap_probe(model, batch, flow_steps=4, max_samples=8):
    """Action-conditioning gaps with common flow noise for low-variance comparison."""
    state = batch["state"].reshape(-1, *batch["state"].shape[-2:])[:max_samples]
    glob = batch["globals"].reshape(-1, batch["globals"].shape[-1])[:max_samples]
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])[:max_samples]
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])[
        :max_samples
    ]
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])[:max_samples]
    opp = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])[
        :max_samples
    ]
    z0 = model.tokenizer.encode(state, glob)
    z1 = model.tokenizer.encode(nxt, nglob)
    state_valid = model.state_token_valid(state)
    noise = model.dynamics.initial_noise_like(z0)
    events, valid, _ = model.action_events(state, action, opp)
    pred = model.dynamics.sample_next(
        z0,
        events,
        valid,
        flow_steps,
        initial_noise=noise,
        state_token_valid=state_valid,
    )
    perm = torch.roll(torch.arange(z0.shape[0], device=z0.device), 1)
    self_events, self_valid, _ = model.action_events(state, action[perm], opp)
    self_pred = model.dynamics.sample_next(
        z0,
        self_events,
        self_valid,
        flow_steps,
        initial_noise=noise,
        state_token_valid=state_valid,
    )
    opp_events, opp_valid, _ = model.action_events(state, action, opp[perm])
    opp_pred = model.dynamics.sample_next(
        z0,
        opp_events,
        opp_valid,
        flow_steps,
        initial_noise=noise,
        state_token_valid=state_valid,
    )
    target = model.dynamics.normalize(z1)
    real = (model.dynamics.normalize(pred) - target).pow(2).mean()
    self_mse = (model.dynamics.normalize(self_pred) - target).pow(2).mean()
    opp_mse = (model.dynamics.normalize(opp_pred) - target).pow(2).mean()
    metrics = {
        "probe/real_mse": real,
        "probe/self_shuffled_mse": self_mse,
        "probe/opp_shuffled_mse": opp_mse,
        "probe/self_gap": self_mse - real,
        "probe/opp_gap": opp_mse - real,
        "probe/self_pred_delta": (
            model.dynamics.normalize(self_pred) - model.dynamics.normalize(pred)
        )
        .pow(2)
        .mean(),
        "probe/opp_pred_delta": (
            model.dynamics.normalize(opp_pred) - model.dynamics.normalize(pred)
        )
        .pow(2)
        .mean(),
    }
    if "counterfactual_valid" in batch:
        cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
        # Search the full batch because only a fraction of rows have paired
        # interventions; the ordinary probes remain capped to max_samples.
        cf_idx = torch.nonzero(cf_valid, as_tuple=False).flatten()[:max_samples]
        if cf_idx.numel():
            cf_departure = batch["state"].reshape(-1, *batch["state"].shape[-2:])[
                cf_idx
            ]
            cf_departure_globals = batch["globals"].reshape(
                -1, batch["globals"].shape[-1]
            )[cf_idx]
            cf_action = batch["counterfactual_action"].reshape(
                -1, *batch["counterfactual_action"].shape[-2:]
            )[cf_idx]
            cf_opp = batch["counterfactual_opponent_action"].reshape(
                -1, *batch["counterfactual_opponent_action"].shape[-2:]
            )[cf_idx]
            cf_state = batch["counterfactual_next_state"].reshape(
                -1, *batch["counterfactual_next_state"].shape[-2:]
            )[cf_idx]
            cf_glob = batch["counterfactual_next_globals"].reshape(
                -1, batch["counterfactual_next_globals"].shape[-1]
            )[cf_idx]
            cf_z0 = model.tokenizer.encode(cf_departure, cf_departure_globals)
            cf_events, cf_event_valid, _ = model.action_events(
                cf_departure, cf_action, cf_opp
            )
            cf_pred = model.dynamics.sample_next(
                cf_z0,
                cf_events,
                cf_event_valid,
                flow_steps,
                initial_noise=model.dynamics.initial_noise_like(cf_z0),
                state_token_valid=model.state_token_valid(cf_departure),
            )
            cf_target = model.tokenizer.encode(cf_state, cf_glob)
            metrics["probe/paired_cf_mse"] = (
                (
                    model.dynamics.normalize(cf_pred)
                    - model.dynamics.normalize(cf_target)
                )
                .pow(2)
                .mean()
            )
    return {key: value.detach() for key, value in metrics.items()}


def _shortcut_consistency(model: StructuredWorldModelV2, z0, z1, events, valid):
    """Dreamer-4 dyadic one-big-step vs two-half-step consistency.

    The big-step base prediction is stop-gradient so this loss specifically
    trains the zero-initialized shortcut skip head.
    """
    wm = model.dynamics
    b = z0.shape[0]
    if b == 0 or wm.emax == 0:
        return z0.sum() * 0.0
    device = z0.device
    e = torch.randint(0, wm.emax, (b,), device=device)
    d = torch.pow(2.0, -e.float())
    cells = torch.pow(2.0, e.float()).long()
    j = (torch.rand(b, device=device) * cells.float()).floor()
    tau = j / cells.float()
    noise = torch.randn_like(z1)
    zt = (1 - tau[:, None, None]) * noise + tau[:, None, None] * z1
    sig = (tau * wm.k_max).round().long().clamp_max(wm.k_max - 1)
    big = wm(z0, zt, events, valid, sig, e)
    big_x = big["base"].detach() + big["skip"]
    big_v = (big_x - zt) / (1 - tau[:, None, None]).clamp_min(1e-4)
    with torch.no_grad():
        ef = e + 1
        p1 = wm(z0, zt, events, valid, sig, ef)["x"]
        v1 = (p1 - zt) / (1 - tau[:, None, None]).clamp_min(1e-4)
        zmid = zt + (d / 2)[:, None, None] * v1
        tau2 = tau + d / 2
        sig2 = (tau2 * wm.k_max).round().long().clamp_max(wm.k_max - 1)
        p2 = wm(z0, zmid, events, valid, sig2, ef)["x"]
        v2 = (p2 - zmid) / (1 - tau2[:, None, None]).clamp_min(1e-4)
        zend = zmid + (d / 2)[:, None, None] * v2
        target_v = (zend - zt) / d[:, None, None]
    return (big_v - target_v).pow(2).mean()


def structured_flow_loss(
    model: StructuredWorldModelV2, batch: dict, structured_coef=1.0, skip_coef=0.25
):
    """Empirical flow + explicit pure-prior + structured grounding + skip."""
    state, glob = batch["state"], batch["globals"]
    nxt, nglob = batch["next_state"], batch["next_globals"]
    state = state.reshape(-1, *state.shape[-2:])
    glob = glob.reshape(-1, glob.shape[-1])
    nxt = nxt.reshape(-1, *nxt.shape[-2:])
    nglob = nglob.reshape(-1, nglob.shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opp = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])
    with torch.no_grad():
        z0 = model.dynamics.normalize(model.tokenizer.encode(state, glob))
        z1 = model.dynamics.normalize(model.tokenizer.encode(nxt, nglob))
    events, valid, overflow = model.action_events(state, action, opp)
    b = z0.shape[0]
    j = torch.randint(0, model.dynamics.k_max, (b,), device=z0.device)
    prior = torch.zeros(b, dtype=torch.bool, device=z0.device)
    nprior = max(1, int(round(b * model.dynamics.cfg.prior_fraction)))
    prior[torch.randperm(b, device=z0.device)[:nprior]] = True
    j[prior] = 0
    tau = j.float() / model.dynamics.k_max
    noise = torch.randn_like(z1)
    zt = (1 - tau[:, None, None]) * noise + tau[:, None, None] * z1
    step = torch.full_like(j, model.dynamics.emax)
    out = model.dynamics(z0, zt, events, valid, j, step)
    per = (out["x"] - z1).pow(2).mean(dim=(1, 2))
    flow = per.mean()
    prior_loss = per[prior].mean()
    decoded = model.tokenizer.decode(model.dynamics.denormalize(out["x"]))
    grounding, gmetrics = structured_reconstruction_loss(
        model.tokenizer, decoded, nxt, nglob
    )
    nskip = int(round(b * model.dynamics.cfg.skip_fraction))
    skip = _shortcut_consistency(
        model, z0[:nskip], z1[:nskip], events[:nskip], valid[:nskip]
    )
    total = flow + prior_loss + structured_coef * grounding + skip_coef * skip
    metrics = {
        "wm/total": total.detach(),
        "flow/matching": flow.detach(),
        "flow/prior": prior_loss.detach(),
        "flow/skip": skip.detach(),
        "action/overflow": overflow.float().mean().detach(),
    }
    metrics.update(gmetrics)
    return total, metrics


def structured_dreamer4_loss(
    model: StructuredWorldModelV2, batch: dict, self_frac: float = 0.25
):
    """Dreamer-4 shortcut forcing on complete-state Markov transitions.

    This is the structured-state counterpart of :func:`shortcut_forcing_loss` in
    ``loss/dreamer.py``. Rows are split between the two paper objectives:

    * empirical x-prediction at the finest step, with the ``0.9 * tau + 0.1``
      signal ramp;
    * one-big-step versus two-half-step bootstrap consistency, converted to
      velocity space and scaled back to x-space by ``(1 - tau) ** 2``.

    Unlike :func:`structured_flow_loss`, this objective does not force or
    double-count pure-prior rows and does not add decoder grounding. The frozen
    structured tokenizer only defines the normalized latent target space.
    """
    if not 0.0 <= self_frac < 1.0:
        raise ValueError(f"self_frac must be in [0, 1), got {self_frac}")

    state, glob = batch["state"], batch["globals"]
    nxt, nglob = batch["next_state"], batch["next_globals"]
    state = state.reshape(-1, *state.shape[-2:])
    glob = glob.reshape(-1, glob.shape[-1])
    nxt = nxt.reshape(-1, *nxt.shape[-2:])
    nglob = nglob.reshape(-1, nglob.shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opp = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])

    wm = model.dynamics
    with torch.no_grad():
        z0 = wm.normalize(model.tokenizer.encode(state, glob))
        z1 = wm.normalize(model.tokenizer.encode(nxt, nglob))
    events, valid, overflow = model.action_events(state, action, opp)

    b = z0.shape[0]
    n_self = int(round(b * self_frac)) if wm.emax > 0 else 0
    n_self = min(n_self, b - 1) if b > 1 else 0
    n_emp = b - n_self
    device = z0.device

    # Empirical rows: x-predict the clean arrival at the finest shortcut step.
    ze0, ze1 = z0[:n_emp], z1[:n_emp]
    ee, ve = events[:n_emp], valid[:n_emp]
    j = torch.randint(0, wm.k_max, (n_emp,), device=device)
    tau = j.float() / wm.k_max
    noise = torch.randn_like(ze1)
    zt = (1 - tau[:, None, None]) * noise + tau[:, None, None] * ze1
    step = torch.full_like(j, wm.emax)
    pred = wm(ze0, zt, ee, ve, j, step)["base"]
    empirical_mse = (pred - ze1).pow(2).mean(dim=(1, 2))
    ramp = 0.9 * tau + 0.1
    empirical = (ramp * empirical_mse).mean()

    metrics = {
        "flow/matching": empirical.detach(),
        "flow/mse": empirical_mse.mean().detach(),
        "action/overflow": overflow.float().mean().detach(),
    }
    for name, selected in (
        ("lo", tau < 1 / 3),
        ("mid", (tau >= 1 / 3) & (tau < 2 / 3)),
        ("hi", tau >= 2 / 3),
    ):
        metrics[f"flow/mse_sigma_{name}"] = (
            empirical_mse[selected].mean().detach()
            if selected.any()
            else empirical_mse.new_zeros(())
        )

    if n_self == 0:
        consistency = empirical.new_zeros(())
        total = empirical
    else:
        # Bootstrap rows: a learned large step must equal two stop-gradient
        # half-steps. The loss is evaluated in velocity space and rescaled to
        # x-space, matching the Dreamer-4 objective.
        zs0, zs1 = z0[n_emp:], z1[n_emp:]
        es, vs = events[n_emp:], valid[n_emp:]
        e = torch.randint(0, wm.emax, (n_self,), device=device)
        d = torch.pow(2.0, -e.float())
        cells = torch.pow(2.0, e.float())
        grid_index = torch.floor(torch.rand(n_self, device=device) * cells)
        tau_s = grid_index / cells
        noise_s = torch.randn_like(zs1)
        zt_s = (1 - tau_s[:, None, None]) * noise_s + tau_s[:, None, None] * zs1
        signal = (tau_s * wm.k_max).round().long()

        with torch.no_grad():
            half_step = e + 1
            first_x = wm(zs0, zt_s, es, vs, signal, half_step)["base"]
            first_v = (first_x - zt_s) / (1 - tau_s[:, None, None]).clamp_min(1e-4)
            zmid = zt_s + (d / 2)[:, None, None] * first_v
            tau_mid = tau_s + d / 2
            signal_mid = (tau_mid * wm.k_max).round().long()
            second_x = wm(zs0, zmid, es, vs, signal_mid, half_step)["base"]
            second_v = (second_x - zmid) / (1 - tau_mid[:, None, None]).clamp_min(1e-4)
            target_v = 0.5 * (first_v + second_v)

        big_x = wm(zs0, zt_s, es, vs, signal, e)["base"]
        big_v = (big_x - zt_s) / (1 - tau_s[:, None, None]).clamp_min(1e-4)
        consistency_per = (big_v - target_v).pow(2).mean(dim=(1, 2))
        consistency_per = consistency_per * (1 - tau_s).pow(2)
        consistency = consistency_per.mean()
        total = (n_emp * empirical + n_self * consistency) / b

    metrics["flow/consistency"] = consistency.detach()
    metrics["flow/total"] = total.detach()
    metrics["wm/total"] = total.detach()
    return total, metrics


def _structured_token_weights(
    tokenizer: StructuredTokenizer,
    states: tuple[torch.Tensor, ...],
    reference_z: torch.Tensor,
    target_z: torch.Tensor,
    *,
    active_boost: float,
    changed_boost: float,
    change_threshold: float,
    padding_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build latent-token weights without rewarding entity padding.

    Spatial tokens always remain valid because empty terrain is still state. An
    entity slot is valid only when at least one state in the comparison has an
    entity at that packed index. Padding receives a small configurable floor so
    it remains decoder-safe without dominating the objective. Occupied spatial
    blocks and real entity slots receive the active boost; tokens whose frozen
    representation changed receive the changed boost.
    """
    b = reference_z.shape[0]
    device = reference_z.device
    ns, ne = tokenizer.n_spatial, tokenizer.n_entity
    valid = torch.ones(b, tokenizer.n_tokens, dtype=torch.bool, device=device)
    active = torch.zeros_like(valid)

    occupied = [state[..., 1].bool() for state in states]
    max_entities = torch.stack([mask.sum(1) for mask in occupied], dim=0).amax(0)
    entity_index = torch.arange(ne, device=device)[None]
    entity_valid = entity_index < max_entities.clamp_max(ne)[:, None]
    valid[:, ns : ns + ne] = entity_valid
    active[:, ns : ns + ne] = entity_valid

    spatial_occupied = torch.stack(occupied, dim=0).any(0)
    ds = tokenizer.cfg.downsample
    spatial_occupied = spatial_occupied.reshape(
        b, tokenizer.h // ds, ds, tokenizer.w // ds, ds
    )
    spatial_occupied = spatial_occupied.any(4).any(2).reshape(b, ns)
    active[:, :ns] = spatial_occupied

    changed = (target_z - reference_z).pow(2).mean(-1) > change_threshold
    changed &= valid
    weights = torch.where(
        valid,
        torch.ones_like(reference_z[..., 0]),
        torch.full_like(reference_z[..., 0], float(padding_weight)),
    )
    weights = weights * (
        1.0
        + float(active_boost) * active.to(weights.dtype)
        + float(changed_boost) * changed.to(weights.dtype)
    )
    denom = valid.sum().clamp_min(1)
    stats = {
        "valid_fraction": valid.float().mean(),
        "active_fraction": (active & valid).sum().float() / denom,
        "changed_fraction": changed.sum().float() / denom,
        "valid": valid,
        "changed": changed,
    }
    return weights, stats


def _weighted_token_mse(prediction, target, weights):
    per_token = (prediction - target).pow(2).mean(-1)
    return (per_token * weights).sum() / weights.sum().clamp_min(1.0)


def _categorical_paired_effect_margin(
    tokenizer: StructuredTokenizer,
    factual_decoded: dict,
    counterfactual_decoded: dict,
    factual_state: torch.Tensor,
    factual_globals: torch.Tensor,
    counterfactual_state: torch.Tensor,
    counterfactual_globals: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Require paired branches to prefer their own exact engine categories.

    Continuous effect geometry can point in the correct direction while both
    branches remain on the same side of a decoder boundary.  This field-wise
    log-probability margin is evaluated only where the cloned engine outcomes
    differ, with the same occupancy/assignment semantics as reconstruction.
    """
    if (
        "numeric_classes" not in factual_decoded
        or "global_classes" not in factual_decoded
    ):
        raise ValueError(
            "categorical paired-effect margin requires an exact-categorical tokenizer"
        )

    losses = []
    effect_count = factual_state.new_zeros((), dtype=torch.float)
    factual_occupied = factual_state[..., 1].bool()
    counterfactual_occupied = counterfactual_state[..., 1].bool()
    factual_assigned = factual_state[..., 7].bool()
    counterfactual_assigned = counterfactual_state[..., 7].bool()

    def add_margin(
        factual_logits,
        counterfactual_logits,
        factual_target,
        cf_target,
        use,
    ):
        nonlocal effect_count
        effect = use & (factual_target != cf_target)
        if not effect.any():
            return
        factual_logp = factual_logits.float().log_softmax(-1)
        cf_logp = counterfactual_logits.float().log_softmax(-1)
        yf, yc = factual_target[..., None], cf_target[..., None]
        factual_preference = (
            factual_logp.gather(-1, yf).squeeze(-1)
            - factual_logp.gather(-1, yc).squeeze(-1)
        )
        cf_preference = (
            cf_logp.gather(-1, yc).squeeze(-1)
            - cf_logp.gather(-1, yf).squeeze(-1)
        )
        losses.append(F.relu(float(margin) - factual_preference[effect]).mean())
        losses.append(F.relu(float(margin) - cf_preference[effect]).mean())
        effect_count = effect_count + effect.sum()

    for name, (index, _size, shift) in tokenizer.categorical.items():
        factual_logits = factual_decoded[name]
        cf_logits = counterfactual_decoded[name]
        yf = (factual_state[..., index] + shift).clamp(
            0, factual_logits.shape[-1] - 1
        ).long()
        yc = (counterfactual_state[..., index] + shift).clamp(
            0, cf_logits.shape[-1] - 1
        ).long()
        if name in ("owner", "unit_type"):
            use = factual_occupied & counterfactual_occupied
        elif name in ("action_type", "direction", "produced_type"):
            use = factual_assigned & counterfactual_assigned
        else:
            use = torch.ones_like(factual_occupied)
        add_margin(factual_logits, cf_logits, yf, yc, use)

    factual_numeric = tokenizer._exact_numeric_values(
        factual_state, factual_globals
    )
    cf_numeric = tokenizer._exact_numeric_values(
        counterfactual_state, counterfactual_globals
    )
    for name, (_index, size, shift, _scale) in tokenizer.numeric_specs.items():
        yf = (factual_numeric[name] + shift).clamp(0, size - 1).long()
        yc = (cf_numeric[name] + shift).clamp(0, size - 1).long()
        use = (
            factual_occupied & counterfactual_occupied
            if name in ("hp", "carried")
            else factual_assigned & counterfactual_assigned
        )
        add_margin(
            factual_decoded["numeric_classes"][name],
            counterfactual_decoded["numeric_classes"][name],
            yf,
            yc,
            use,
        )

    for name, (index, size, shift, _scale) in tokenizer.global_specs.items():
        yf = (factual_globals[..., index] + shift).clamp(0, size - 1).long()
        yc = (counterfactual_globals[..., index] + shift).clamp(0, size - 1).long()
        add_margin(
            factual_decoded["global_classes"][name],
            counterfactual_decoded["global_classes"][name],
            yf,
            yc,
            torch.ones_like(yf, dtype=torch.bool),
        )

    if not losses:
        return factual_decoded["present"].new_zeros(()), effect_count
    return torch.stack(losses).mean(), effect_count


def structured_causal_paired_loss(
    model: StructuredWorldModelV2,
    batch: dict,
    *,
    factual_coef: float = 1.0,
    counterfactual_coef: float = 1.0,
    effect_coef: float = 1.0,
    active_token_boost: float = 1.0,
    changed_token_boost: float = 4.0,
    change_threshold: float = 1e-6,
    padding_token_weight: float = 0.0,
    effect_cosine_coef: float = 0.0,
    effect_norm_coef: float = 0.0,
    canonical_grounding_coef: float = 0.0,
    canonical_changed_boost: float = 0.0,
    canonical_effect_margin_coef: float = 0.0,
    canonical_effect_margin: float = 1.0,
    rollout_grounding_coef: float = 0.0,
    rollout_latent_coef: float = 0.0,
    rollout_horizon: int = 1,
    rollout_discount: float = 1.0,
    rollout_batch_fraction: float = 1.0,
    residual_correction_coef: float = 0.0,
):
    """Direct one-step dynamics with paired counterfactual effect alignment.

    The deployed one-step sampler calls the dynamics model at ``tau=0, d=1``.
    This objective trains that exact query rather than allowing a clean noised
    arrival to leak the target at high tau. Factual and counterfactual branches
    share their noise, and their predicted difference is explicitly aligned to
    the frozen-tokenizer action effect.
    """
    if padding_token_weight < 0.0:
        raise ValueError("padding_token_weight must be non-negative")
    if residual_correction_coef < 0.0:
        raise ValueError("residual_correction_coef must be non-negative")
    if residual_correction_coef > 0.0 and not model.dynamics.cfg.residual_prediction:
        raise ValueError(
            "residual_correction_coef requires residual_prediction=true"
        )
    if canonical_effect_margin_coef < 0.0:
        raise ValueError("canonical_effect_margin_coef must be non-negative")
    if rollout_grounding_coef < 0.0 or rollout_latent_coef < 0.0:
        raise ValueError("rollout coefficients must be non-negative")
    if rollout_horizon < 1:
        raise ValueError("rollout_horizon must be positive")
    if not 0.0 < rollout_batch_fraction <= 1.0:
        raise ValueError("rollout_batch_fraction must be in (0, 1]")
    if not 0.0 < rollout_discount <= 1.0:
        raise ValueError("rollout_discount must be in (0, 1]")
    required = (
        "counterfactual_action",
        "counterfactual_opponent_action",
        "counterfactual_next_state",
        "counterfactual_next_globals",
        "counterfactual_valid",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(
            "causal_paired objective requires counterfactual dataset fields; "
            f"missing {missing}"
        )

    state_sequence = batch["state"]
    global_sequence = batch["globals"]
    next_state_sequence = batch["next_state"]
    next_global_sequence = batch["next_globals"]
    action_sequence = batch["action"]
    opponent_sequence = batch["opponent_action"]
    sequence_batch, sequence_length = state_sequence.shape[:2]
    state = state_sequence.reshape(-1, *state_sequence.shape[-2:])
    glob = global_sequence.reshape(-1, global_sequence.shape[-1])
    nxt = batch["next_state"].reshape(-1, *batch["next_state"].shape[-2:])
    nglob = batch["next_globals"].reshape(-1, batch["next_globals"].shape[-1])
    action = batch["action"].reshape(-1, *batch["action"].shape[-2:])
    opp = batch["opponent_action"].reshape(-1, *batch["opponent_action"].shape[-2:])
    wm = model.dynamics
    with torch.no_grad():
        z0 = wm.normalize(model.tokenizer.encode(state, glob))
        z1 = wm.normalize(model.tokenizer.encode(nxt, nglob))

    events, valid, overflow = model.action_events(state, action, opp)
    noise = wm.initial_noise_like(z1)
    signal = torch.zeros(z0.shape[0], dtype=torch.long, device=z0.device)
    step = torch.zeros_like(signal)
    state_token_valid = model.state_token_valid(state)
    factual_out = wm(
        z0,
        noise,
        events,
        valid,
        signal,
        step,
        state_token_valid=state_token_valid,
    )
    factual_pred = factual_out["base"]
    correction_regularizer = factual_out["correction"].pow(2).mean()
    factual_weights, factual_stats = _structured_token_weights(
        model.tokenizer,
        (state, nxt),
        z0,
        z1,
        active_boost=active_token_boost,
        changed_boost=changed_token_boost,
        change_threshold=change_threshold,
        padding_weight=padding_token_weight,
    )
    factual = _weighted_token_mse(factual_pred, z1, factual_weights)
    copy_mse = _weighted_token_mse(z0, z1, factual_weights)
    factual_per_token = (factual_pred - z1).pow(2).mean(-1)
    copy_per_token = (z0 - z1).pow(2).mean(-1)
    factual_valid = factual_stats["valid"]
    factual_padding = ~factual_valid
    valid_mse = factual_per_token[factual_valid].mean()
    valid_copy_mse = copy_per_token[factual_valid].mean()
    padding_mse = (
        factual_per_token[factual_padding].mean()
        if factual_padding.any()
        else factual.new_zeros(())
    )

    cf_valid = batch["counterfactual_valid"].reshape(-1).bool()
    zero = factual.new_zeros(())
    counterfactual = zero
    effect = zero
    effect_cosine = zero
    effect_norm_ratio = zero
    effect_norm_ratio_aggregate = zero
    effect_cosine_loss = zero
    effect_norm_loss = zero
    effect_cosine_sum = zero
    effect_geometry_rows = zero
    effect_predicted_norm_sum = zero
    effect_target_norm_sum = zero
    effect_token_fraction = zero
    cf_overflow = zero
    cf_pred = None
    cf_state = None
    cf_glob = None
    if cf_valid.any():
        idx = torch.nonzero(cf_valid, as_tuple=False).flatten()
        raw_cf_state = batch["counterfactual_next_state"].reshape(
            -1, *batch["counterfactual_next_state"].shape[-2:]
        )
        raw_cf_glob = batch["counterfactual_next_globals"].reshape(
            -1, batch["counterfactual_next_globals"].shape[-1]
        )
        # Encode factual and counterfactual arrivals at the same batch shape.
        # Sub-batch SDPA kernels can differ by roundoff even for identical input,
        # which otherwise turns engine-zero effects into fake latent effects.
        cf_state_all = torch.where(
            cf_valid[:, None, None], raw_cf_state, nxt
        )
        cf_glob_all = torch.where(cf_valid[:, None], raw_cf_glob, nglob)
        cf_state = cf_state_all[idx]
        cf_glob = cf_glob_all[idx]
        cf_action = batch["counterfactual_action"].reshape(
            -1, *batch["counterfactual_action"].shape[-2:]
        )[idx]
        cf_opp = batch["counterfactual_opponent_action"].reshape(
            -1, *batch["counterfactual_opponent_action"].shape[-2:]
        )[idx]
        with torch.no_grad():
            zcf_all = wm.normalize(
                model.tokenizer.encode(cf_state_all, cf_glob_all)
            )
            zcf = zcf_all[idx]
        cf_events, cf_event_valid, cf_overflow_rows = model.action_events(
            state[idx], cf_action, cf_opp
        )
        cf_out = wm(
            z0[idx],
            noise[idx],
            cf_events,
            cf_event_valid,
            signal[idx],
            step[idx],
            state_token_valid=state_token_valid[idx],
        )
        cf_pred = cf_out["base"]
        correction_regularizer = correction_regularizer + cf_out[
            "correction"
        ].pow(2).mean()
        cf_weights, _ = _structured_token_weights(
            model.tokenizer,
            (state[idx], cf_state),
            z0[idx],
            zcf,
            active_boost=active_token_boost,
            changed_boost=changed_token_boost,
            change_threshold=change_threshold,
            padding_weight=padding_token_weight,
        )
        counterfactual = _weighted_token_mse(cf_pred, zcf, cf_weights)

        target_effect = zcf - z1[idx]
        predicted_effect = cf_pred - factual_pred[idx]
        effect_weights, effect_stats = _structured_token_weights(
            model.tokenizer,
            (state[idx], nxt[idx], cf_state),
            z1[idx],
            zcf,
            active_boost=active_token_boost,
            changed_boost=changed_token_boost,
            change_threshold=change_threshold,
            padding_weight=padding_token_weight,
        )
        effect = _weighted_token_mse(predicted_effect, target_effect, effect_weights)
        effect_token_fraction = effect_stats["changed_fraction"]

        # Report the geometry over valid tokens, while excluding paired rows for
        # which the alternative action happened to have no state effect.
        geometry_mask = effect_stats["valid"].to(target_effect.dtype)[..., None]
        target_geometry = (target_effect * geometry_mask).flatten(1)
        predicted_geometry = (predicted_effect * geometry_mask).flatten(1)
        target_norm = target_geometry.norm(dim=1)
        has_effect = target_norm > 1e-8
        if has_effect.any():
            predicted_norm = predicted_geometry[has_effect].norm(dim=1)
            cosines = F.cosine_similarity(
                predicted_geometry[has_effect], target_geometry[has_effect], dim=1
            )
            effect_cosine = cosines.mean()
            effect_norm_ratio = (
                predicted_norm / target_norm[has_effect].clamp_min(1e-8)
            ).mean()
            effect_norm_ratio_aggregate = predicted_norm.sum() / target_norm[
                has_effect
            ].sum().clamp_min(1e-8)
            effect_cosine_loss = (1.0 - cosines).mean()
            predicted_norm_sum = predicted_norm.sum()
            target_norm_sum = target_norm[has_effect].sum()
            # Optimize one global log-ratio, not a mean of per-row ratios. Tiny
            # legitimate effects otherwise create unbounded gradients and the
            # same spikes that made the old validation metric misleading.
            log_norm_ratio = torch.log(
                (predicted_norm_sum + 1e-6) / (target_norm_sum + 1e-6)
            )
            effect_norm_loss = F.smooth_l1_loss(
                log_norm_ratio, torch.zeros_like(log_norm_ratio)
            )
            effect_cosine_sum = cosines.sum()
            effect_geometry_rows = has_effect.sum().to(factual.dtype)
            effect_predicted_norm_sum = predicted_norm_sum
            effect_target_norm_sum = target_norm_sum
        cf_overflow = cf_overflow_rows.float().mean()

    grounding_factual = zero
    grounding_counterfactual = zero
    categorical_effect_margin = zero
    categorical_effect_fields = zero
    grounding_metrics = {}
    factual_decoded = None
    cf_decoded = None
    if canonical_grounding_coef > 0.0 or canonical_effect_margin_coef > 0.0:
        # The frozen decoder is differentiable with respect to predicted latents.
        # Canonical field losses prevent a small latent error from crossing a
        # categorical unit/assignment boundary unnoticed.
        factual_changed = state.clone()
        factual_target = nxt.clone()
        factual_changed[..., 2] = -1
        factual_target[..., 2] = -1
        factual_changed = (factual_changed != factual_target).any(-1)
        factual_cell_weights = 1.0 + float(canonical_changed_boost) * factual_changed
        factual_decoded = model.tokenizer.decode(wm.denormalize(factual_pred))
        if canonical_grounding_coef > 0.0:
            grounding_factual, factual_ground_metrics = structured_reconstruction_loss(
                model.tokenizer,
                factual_decoded,
                nxt,
                nglob,
                cell_weights=factual_cell_weights,
            )
            selected_metrics = (
                "present_acc",
                "unit_type_acc",
                "exact_cell",
                "exact_frame",
                "exact_globals",
                "exact_roundtrip",
                "exact_occupied_cell",
                "exact_assigned_cell",
                "total",
            )
            grounding_metrics.update(
                {
                    f"grounding/factual_{name}": factual_ground_metrics[f"tok/{name}"]
                    for name in selected_metrics
                }
            )
            grounding_metrics["grounding/factual_dynamics_score"] = torch.stack(
                (
                    factual_ground_metrics["tok/exact_occupied_cell"],
                    factual_ground_metrics["tok/exact_assigned_cell"],
                    factual_ground_metrics["tok/exact_globals"],
                    factual_ground_metrics["tok/exact_frame"],
                )
            ).mean()
        if cf_pred is not None:
            cf_before = state[idx].clone()
            cf_target = cf_state.clone()
            cf_before[..., 2] = -1
            cf_target[..., 2] = -1
            cf_changed = (cf_before != cf_target).any(-1)
            cf_cell_weights = 1.0 + float(canonical_changed_boost) * cf_changed
            cf_decoded = model.tokenizer.decode(wm.denormalize(cf_pred))
            if canonical_grounding_coef > 0.0:
                grounding_counterfactual, cf_ground_metrics = (
                    structured_reconstruction_loss(
                        model.tokenizer,
                        cf_decoded,
                        cf_state,
                        cf_glob,
                        cell_weights=cf_cell_weights,
                    )
                )
                grounding_metrics.update(
                    {
                        f"grounding/counterfactual_{name}": cf_ground_metrics[
                            f"tok/{name}"
                        ]
                        for name in selected_metrics
                    }
                )
            if canonical_effect_margin_coef > 0.0:
                categorical_effect_margin, categorical_effect_fields = (
                    _categorical_paired_effect_margin(
                        model.tokenizer,
                        {
                            key: (
                                {
                                    subkey: value[idx]
                                    for subkey, value in values.items()
                                }
                                if isinstance(values, dict)
                                else values[idx]
                            )
                            for key, values in factual_decoded.items()
                        },
                        cf_decoded,
                        nxt[idx],
                        nglob[idx],
                        cf_state,
                        cf_glob,
                        margin=canonical_effect_margin,
                    )
                )
    grounding = grounding_factual + grounding_counterfactual

    # Observation overshooting: start from the model's own first prediction and
    # repeatedly apply the deployed tau=0,d=1 transition.  The recorded actions
    # are routed against the model-decoded departure state, matching inference.
    rollout_grounding = zero
    rollout_latent = zero
    rollout_exact_cell = zero
    rollout_exact_frame = zero
    rollout_exact_globals = zero
    rollout_steps = 0
    effective_horizon = min(int(rollout_horizon), int(sequence_length))
    if (
        effective_horizon > 1
        and (rollout_grounding_coef > 0.0 or rollout_latent_coef > 0.0)
    ):
        rows = max(1, int(round(sequence_batch * float(rollout_batch_fraction))))
        row_index = torch.arange(rows, device=state.device)
        factual_sequence = factual_pred.reshape(
            sequence_batch, sequence_length, *factual_pred.shape[-2:]
        )
        target_z_sequence = z1.reshape(
            sequence_batch, sequence_length, *z1.shape[-2:]
        )
        rollout_z = factual_sequence[row_index, 0]
        rollout_weights_sum = zero
        rollout_correction = []
        for time_index in range(1, effective_horizon):
            with torch.no_grad():
                rollout_state, _ = model.tokenizer.discretize(
                    model.tokenizer.decode(wm.denormalize(rollout_z))
                )
            events_t, valid_t, _ = model.action_events(
                rollout_state,
                action_sequence[row_index, time_index],
                opponent_sequence[row_index, time_index],
            )
            noise_t = wm.initial_noise_like(rollout_z)
            zeros_t = torch.zeros(rows, dtype=torch.long, device=state.device)
            rollout_out = wm(
                rollout_z,
                noise_t,
                events_t,
                valid_t,
                zeros_t,
                zeros_t,
                state_token_valid=model.state_token_valid(rollout_state),
            )
            rollout_z = rollout_out["base"]
            rollout_correction.append(rollout_out["correction"].pow(2).mean())
            target_state_t = next_state_sequence[row_index, time_index]
            target_globals_t = next_global_sequence[row_index, time_index]
            target_z_t = target_z_sequence[row_index, time_index]
            discount = rollout_z.new_tensor(
                float(rollout_discount) ** (time_index - 1)
            )
            rollout_weights_sum = rollout_weights_sum + discount
            if rollout_latent_coef > 0.0:
                token_weights, _ = _structured_token_weights(
                    model.tokenizer,
                    (rollout_state, target_state_t),
                    rollout_z,
                    target_z_t,
                    active_boost=active_token_boost,
                    changed_boost=changed_token_boost,
                    change_threshold=change_threshold,
                    padding_weight=padding_token_weight,
                )
                rollout_latent = rollout_latent + discount * _weighted_token_mse(
                    rollout_z, target_z_t, token_weights
                )
            if rollout_grounding_coef > 0.0:
                true_source_t = state_sequence[row_index, time_index]
                source_cmp, target_cmp = true_source_t.clone(), target_state_t.clone()
                source_cmp[..., 2] = -1
                target_cmp[..., 2] = -1
                changed_t = (source_cmp != target_cmp).any(-1)
                cell_weights_t = (
                    1.0 + float(canonical_changed_boost) * changed_t
                )
                grounding_t, metrics_t = structured_reconstruction_loss(
                    model.tokenizer,
                    model.tokenizer.decode(wm.denormalize(rollout_z)),
                    target_state_t,
                    target_globals_t,
                    cell_weights=cell_weights_t,
                )
                rollout_grounding = rollout_grounding + discount * grounding_t
                rollout_exact_cell = (
                    rollout_exact_cell + discount * metrics_t["tok/exact_cell"]
                )
                rollout_exact_frame = (
                    rollout_exact_frame + discount * metrics_t["tok/exact_frame"]
                )
                rollout_exact_globals = (
                    rollout_exact_globals + discount * metrics_t["tok/exact_globals"]
                )
            rollout_steps += 1
        rollout_weights_sum = rollout_weights_sum.clamp_min(1.0e-8)
        rollout_grounding = rollout_grounding / rollout_weights_sum
        rollout_latent = rollout_latent / rollout_weights_sum
        rollout_exact_cell = rollout_exact_cell / rollout_weights_sum
        rollout_exact_frame = rollout_exact_frame / rollout_weights_sum
        rollout_exact_globals = rollout_exact_globals / rollout_weights_sum
        if rollout_correction:
            correction_regularizer = correction_regularizer + torch.stack(
                rollout_correction
            ).mean()

    total = (
        float(factual_coef) * factual
        + float(counterfactual_coef) * counterfactual
        + float(effect_coef) * effect
        + float(effect_cosine_coef) * effect_cosine_loss
        + float(effect_norm_coef) * effect_norm_loss
        + float(canonical_grounding_coef) * grounding
        + float(canonical_effect_margin_coef) * categorical_effect_margin
        + float(rollout_grounding_coef) * rollout_grounding
        + float(rollout_latent_coef) * rollout_latent
        + float(residual_correction_coef) * correction_regularizer
    )
    metrics = {
        "wm/total": total.detach(),
        "flow/matching": factual.detach(),
        "flow/total": total.detach(),
        "causal/factual": factual.detach(),
        "causal/counterfactual": counterfactual.detach(),
        "causal/effect": effect.detach(),
        "causal/effect_cosine_loss": effect_cosine_loss.detach(),
        "causal/effect_norm_loss": effect_norm_loss.detach(),
        "causal/copy_mse": copy_mse.detach(),
        "causal/valid_mse": valid_mse.detach(),
        "causal/valid_copy_mse": valid_copy_mse.detach(),
        "causal/padding_mse": padding_mse.detach(),
        "causal/unweighted_mse": (factual_pred - z1).pow(2).mean().detach(),
        "causal/effect_cosine": effect_cosine.detach(),
        "causal/effect_norm_ratio": effect_norm_ratio.detach(),
        "causal/effect_norm_ratio_aggregate": (
            effect_norm_ratio_aggregate.detach()
        ),
        "causal/effect_cosine_sum": effect_cosine_sum.detach(),
        "causal/effect_geometry_rows": effect_geometry_rows.detach(),
        "causal/effect_predicted_norm_sum": effect_predicted_norm_sum.detach(),
        "causal/effect_target_norm_sum": effect_target_norm_sum.detach(),
        "causal/grounding_factual": grounding_factual.detach(),
        "causal/grounding_counterfactual": grounding_counterfactual.detach(),
        "causal/categorical_effect_margin": categorical_effect_margin.detach(),
        "causal/categorical_effect_fields": categorical_effect_fields.detach(),
        "causal/correction_regularizer": correction_regularizer.detach(),
        "causal/cf_valid_fraction": cf_valid.float().mean().detach(),
        "causal/effect_token_fraction": effect_token_fraction.detach(),
        "causal/valid_token_fraction": factual_stats["valid_fraction"].detach(),
        "causal/active_token_fraction": factual_stats["active_fraction"].detach(),
        "causal/changed_token_fraction": factual_stats["changed_fraction"].detach(),
        "action/overflow": overflow.float().mean().detach(),
        "action/counterfactual_overflow": cf_overflow.detach(),
        "rollout/grounding": rollout_grounding.detach(),
        "rollout/latent": rollout_latent.detach(),
        "rollout/exact_cell": rollout_exact_cell.detach(),
        "rollout/exact_frame": rollout_exact_frame.detach(),
        "rollout/exact_globals": rollout_exact_globals.detach(),
        "rollout/steps": factual.new_tensor(float(rollout_steps)).detach(),
    }
    metrics.update(grounding_metrics)
    return total, metrics
