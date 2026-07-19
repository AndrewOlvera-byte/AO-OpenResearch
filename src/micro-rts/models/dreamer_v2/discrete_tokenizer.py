"""Hard product-code tokenizer for canonical MicroRTS engine state.

The continuous structured tokenizer remains available for Dreamer/flow models.
This module is a parallel, strictly discrete interface: canonical state is
encoded to integer product codes, and every reconstruction head receives only
the corresponding hard code embeddings.  There is no continuous bypass.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiscreteTokenizerConfig
from .schema import GLOBAL_WIDTH, STATE_WIDTH


class DiscreteStructuredTokenizer(nn.Module):
    """Canonical state <-> hard categorical product codes."""

    def __init__(
        self,
        grid_hw=(16, 16),
        cfg: DiscreteTokenizerConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or DiscreteTokenizerConfig()
        self.h, self.w = map(int, grid_hw)
        ds = int(self.cfg.spatial_downsample)
        if ds < 1 or self.h % ds or self.w % ds:
            raise ValueError("spatial_downsample must divide both grid dimensions")
        if self.cfg.codebook_depth < 1 or self.cfg.codebook_size < 2:
            raise ValueError("the discrete bottleneck needs positive depth and size")

        d = int(self.cfg.d_model)
        self.n_spatial = (self.h // ds) * (self.w // ds)
        self.n_global = int(self.cfg.n_global_tokens)
        self.n_tokens = self.n_spatial + self.n_global
        self.codebook_depth = int(self.cfg.codebook_depth)
        self.codebook_size = int(self.cfg.codebook_size)
        self.n_code_tokens = self.n_tokens * self.codebook_depth
        self.d_latent = d

        # Unit IDs are JVM allocation counters and intentionally excluded.
        self.cell_specs = {
            "terrain": (0, 3, 0),
            "present": (1, 2, 0),
            "owner": (3, 4, 1),
            "unit_type": (4, self.cfg.max_unit_types + 1, 1),
            "hp": (5, self.cfg.max_hp + 1, 0),
            "carried": (6, self.cfg.max_carried + 1, 0),
            "assignment": (7, 2, 0),
            "action_type": (8, 7, 1),
            "direction": (9, 6, 1),
            "target_x": (10, self.w + 1, 1),
            "target_y": (11, self.h + 1, 1),
            "produced_type": (12, self.cfg.max_unit_types + 1, 1),
            "start_tick": (13, self.cfg.max_tick + 2, 1),
            "eta": (14, self.cfg.max_eta + 1, 0),
            "remaining": (15, self.cfg.max_remaining + 1, 0),
        }
        self.global_specs = {
            "tick": (0, self.cfg.max_tick + 1, 0),
            "self_resources": (1, self.cfg.max_resources + 1, 0),
            "opponent_resources": (2, self.cfg.max_resources + 1, 0),
            "self_reserved": (3, self.cfg.max_resources + 1, 0),
            "opponent_reserved": (4, self.cfg.max_resources + 1, 0),
            "reserved_positions": (5, self.cfg.max_reserved_positions + 1, 0),
            "winner": (6, 4, 1),
            "gameover": (7, 2, 0),
        }

        self.cell_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, d)
                for name, (_, size, _) in self.cell_specs.items()
            }
        )
        self.global_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(size, d)
                for name, (_, size, _) in self.global_specs.items()
            }
        )
        self.x_embed = nn.Embedding(self.w, d)
        self.y_embed = nn.Embedding(self.h, d)
        factor = ds * ds
        self.patch_in = nn.Linear(factor * d, d)
        layer = nn.TransformerEncoderLayer(
            d,
            self.cfg.n_heads,
            int(self.cfg.mlp_ratio * d),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(
            layer, self.cfg.depth, nn.LayerNorm(d)
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_WIDTH * d, 2 * d),
            nn.SiLU(),
            nn.Linear(2 * d, self.n_global * d),
        )

        self.code_logits = nn.Linear(d, self.codebook_depth * self.codebook_size)
        self.codebooks = nn.Parameter(
            torch.empty(self.codebook_depth, self.codebook_size, d)
        )
        nn.init.normal_(self.codebooks, std=1.0 / math.sqrt(d))
        spatial_h, spatial_w = self.h // ds, self.w // ds
        semantic = torch.arange(self.n_tokens)
        is_spatial = semantic < self.n_spatial
        self.register_buffer(
            "code_x",
            torch.where(is_spatial, semantic % spatial_w, spatial_w),
            persistent=False,
        )
        self.register_buffer(
            "code_y",
            torch.where(is_spatial, semantic // spatial_w, spatial_h),
            persistent=False,
        )
        self.register_buffer("code_type", (~is_spatial).long(), persistent=False)
        self.code_x_embed = nn.Embedding(spatial_w + 1, d)
        self.code_y_embed = nn.Embedding(spatial_h + 1, d)
        self.code_type_embed = nn.Embedding(2, d)
        self.code_semantic_embed = nn.Embedding(self.n_tokens, d)
        self.code_depth_embed = nn.Embedding(self.codebook_depth, d)
        for embedding in (
            self.code_x_embed,
            self.code_y_embed,
            self.code_type_embed,
            self.code_semantic_embed,
            self.code_depth_embed,
        ):
            nn.init.normal_(embedding.weight, std=0.02)

        self.patch_out = nn.Linear(d, factor * d)
        self.global_context = nn.Sequential(
            nn.Linear(self.n_global * d, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.cell_heads = nn.ModuleDict(
            {name: nn.Linear(d, size) for name, (_, size, _) in self.cell_specs.items()}
        )
        self.global_heads = nn.ModuleDict(
            {
                name: nn.Linear(self.n_global * d, size)
                for name, (_, size, _) in self.global_specs.items()
            }
        )
        self.legacy_head = nn.Linear(d, self.cfg.legacy_obs_channels)
        self.mask_head = nn.Linear(d, self.cfg.mask_width)

    def _flatten(self, state, globals_):
        lead = state.shape[:-2]
        return (
            state.reshape(-1, self.h * self.w, STATE_WIDTH),
            globals_.reshape(-1, GLOBAL_WIDTH),
            lead,
        )

    @staticmethod
    def _safe_ids(values: torch.Tensor, size: int, shift: int) -> torch.Tensor:
        return (values.long() + int(shift)).clamp(0, int(size) - 1)

    def _code_positions(self) -> torch.Tensor:
        semantic = (
            self.code_x_embed(self.code_x)
            + self.code_y_embed(self.code_y)
            + self.code_type_embed(self.code_type)
            + self.code_semantic_embed.weight
        )
        return semantic[:, None] + self.code_depth_embed.weight[None]

    def _encode_hidden(self, state, globals_):
        state, globals_, lead = self._flatten(state, globals_)
        cell = 0.0
        for name, (index, size, shift) in self.cell_specs.items():
            cell = cell + self.cell_embeddings[name](
                self._safe_ids(state[..., index], size, shift)
            )
        cell = cell / math.sqrt(len(self.cell_specs))
        x = torch.arange(self.w, device=state.device).repeat(self.h)
        y = torch.arange(self.h, device=state.device).repeat_interleave(self.w)
        cell = cell + self.x_embed(x)[None] + self.y_embed(y)[None]

        ds, d = self.cfg.spatial_downsample, self.cfg.d_model
        spatial = cell.reshape(-1, self.h // ds, ds, self.w // ds, ds, d)
        spatial = spatial.permute(0, 1, 3, 2, 4, 5).reshape(
            -1, self.n_spatial, ds * ds * d
        )
        spatial = self.spatial_encoder(self.patch_in(spatial))

        global_parts = []
        for name, (index, size, shift) in self.global_specs.items():
            global_parts.append(
                self.global_embeddings[name](
                    self._safe_ids(globals_[..., index], size, shift)
                )
            )
        global_tokens = self.global_encoder(torch.cat(global_parts, -1)).reshape(
            -1, self.n_global, d
        )
        hidden = torch.cat((spatial, global_tokens), dim=1)
        return hidden.reshape(*lead, self.n_tokens, d), lead

    def _quantize(self, hidden: torch.Tensor):
        lead = hidden.shape[:-2]
        logits = self.code_logits(hidden).reshape(
            *lead, self.n_tokens, self.codebook_depth, self.codebook_size
        )
        probs = logits.softmax(-1)
        ids = probs.argmax(-1)
        hard = F.one_hot(ids, self.codebook_size).to(probs.dtype)
        assignments = hard + probs - probs.detach() if self.training else hard
        quantized_depth = torch.einsum(
            "...ndk,dkh->...ndh", assignments, self.codebooks
        )
        quantized_depth = quantized_depth + self._code_positions()
        quantized = quantized_depth.sum(-2) / math.sqrt(self.codebook_depth)
        return quantized, ids, logits, probs

    def encode_codes(self, state, globals_) -> torch.Tensor:
        hidden, _ = self._encode_hidden(state, globals_)
        return self._quantize(hidden)[1]

    def embed_code_tokens(
        self, codes: torch.Tensor, *, include_position: bool = True
    ) -> torch.Tensor:
        """Embed each base code separately, returning ``[..., N*D, d]``."""
        if codes.shape[-2:] != (self.n_tokens, self.codebook_depth):
            raise ValueError(
                f"code tail {tuple(codes.shape[-2:])} != "
                f"{(self.n_tokens, self.codebook_depth)}"
            )
        lead = codes.shape[:-2]
        flat = codes.reshape(-1, self.n_tokens, self.codebook_depth)
        depth = torch.arange(self.codebook_depth, device=codes.device)
        embedded = self.codebooks[depth[None, None, :], flat]
        if include_position:
            embedded = embedded + self._code_positions()[None]
        return embedded.reshape(*lead, self.n_code_tokens, self.d_latent)

    def semantic_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        tokens = self.embed_code_tokens(codes)
        return tokens.reshape(
            *codes.shape[:-2], self.n_tokens, self.codebook_depth, self.d_latent
        ).sum(-2) / math.sqrt(self.codebook_depth)

    def decode_embeddings(self, quantized: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = quantized.shape[:-2]
        q = quantized.reshape(-1, self.n_tokens, self.d_latent)
        spatial = q[:, : self.n_spatial]
        global_tokens = q[:, self.n_spatial :]
        ds, d = self.cfg.spatial_downsample, self.d_latent
        cell = self.patch_out(spatial).reshape(
            -1, self.h // ds, self.w // ds, ds, ds, d
        )
        cell = cell.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.h * self.w, d)
        global_flat = global_tokens.flatten(1)
        cell = cell + self.global_context(global_flat)[:, None]
        decoded = {
            name: head(cell).reshape(*lead, self.h * self.w, -1)
            for name, head in self.cell_heads.items()
        }
        decoded["globals"] = {
            name: head(global_flat).reshape(*lead, -1)
            for name, head in self.global_heads.items()
        }
        decoded["legacy_obs"] = self.legacy_head(cell).reshape(
            *lead, self.h * self.w, -1
        )
        decoded["mask"] = self.mask_head(cell).reshape(*lead, self.h * self.w, -1)
        return decoded

    def decode_codes(self, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.decode_embeddings(self.semantic_embeddings(codes))

    def forward(self, state, globals_):
        hidden, _ = self._encode_hidden(state, globals_)
        quantized, codes, logits, probs = self._quantize(hidden)
        return (
            self.decode_embeddings(quantized),
            codes,
            {
                "code_logits": logits,
                "code_probs": probs,
            },
        )

    def discretize(self, decoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
        lead = decoded["present"].shape[:-2]
        state = torch.zeros(
            *lead,
            self.h * self.w,
            STATE_WIDTH,
            dtype=torch.long,
            device=decoded["present"].device,
        )
        state[..., 2] = -1
        for name, (index, _size, shift) in self.cell_specs.items():
            state[..., index] = decoded[name].argmax(-1) - shift
        globals_ = torch.zeros(
            *lead, GLOBAL_WIDTH, dtype=torch.long, device=state.device
        )
        for name, (index, _size, shift) in self.global_specs.items():
            globals_[..., index] = decoded["globals"][name].argmax(-1) - shift

        empty = state[..., 1] == 0
        for index in (2, 3, 4, 8, 9, 10, 11, 12, 13):
            state[..., index] = torch.where(
                empty, state[..., index].new_full((), -1), state[..., index]
            )
        state[..., 5:7] = torch.where(
            empty[..., None], torch.zeros_like(state[..., 5:7]), state[..., 5:7]
        )
        state[..., 7] = torch.where(
            empty, torch.zeros_like(state[..., 7]), state[..., 7]
        )
        unassigned = state[..., 7] == 0
        for index in (8, 9, 10, 11, 12, 13):
            state[..., index] = torch.where(
                unassigned, state[..., index].new_full((), -1), state[..., index]
            )
        state[..., 14:16] = torch.where(
            unassigned[..., None],
            torch.zeros_like(state[..., 14:16]),
            state[..., 14:16],
        )
        return state, globals_


def _masked_ce(logits, target, use):
    per = F.cross_entropy(
        logits.flatten(0, -2), target.flatten(), reduction="none"
    ).reshape_as(target)
    weight = use.to(per.dtype)
    return (per * weight).sum() / weight.sum().clamp_min(1.0)


def discrete_reconstruction_loss(
    tokenizer: DiscreteStructuredTokenizer,
    decoded: dict,
    state: torch.Tensor,
    globals_: torch.Tensor,
    code_probs: torch.Tensor,
    legacy_obs: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    *,
    usage_coef: float = 0.01,
):
    """Categorical reconstruction plus product-code usage regularization."""
    occupied = state[..., 1].bool()
    assigned = state[..., 7].bool()
    losses: dict[str, torch.Tensor] = {}
    metrics: dict[str, torch.Tensor] = {}
    oov = state.new_zeros((), dtype=torch.float)

    for name, (index, size, shift) in tokenizer.cell_specs.items():
        raw = state[..., index] + shift
        oov = oov + ((raw < 0) | (raw >= size)).float().mean()
        target = raw.clamp(0, size - 1).long()
        if name in ("owner", "unit_type", "hp", "carried"):
            use = occupied
        elif name in (
            "action_type",
            "direction",
            "target_x",
            "target_y",
            "produced_type",
            "start_tick",
            "eta",
            "remaining",
        ):
            use = assigned
        else:
            use = torch.ones_like(occupied)
        losses[name] = _masked_ce(decoded[name], target, use)
        pred = decoded[name].argmax(-1)
        metrics[f"dtok/{name}_acc"] = (
            ((pred == target) & use).sum().float() / use.sum().clamp_min(1)
        ).detach()

    for name, (index, size, shift) in tokenizer.global_specs.items():
        raw = globals_[..., index] + shift
        oov = oov + ((raw < 0) | (raw >= size)).float().mean()
        target = raw.clamp(0, size - 1).long()
        global_logits = decoded["globals"][name]
        losses[f"global_{name}"] = F.cross_entropy(
            global_logits.flatten(0, -2), target.flatten()
        )
        metrics[f"dtok/global_{name}_acc"] = (
            (global_logits.argmax(-1) == target).float().mean().detach()
        )

    if legacy_obs is not None:
        target = legacy_obs.movedim(-3, -1).reshape(
            *legacy_obs.shape[:-3], tokenizer.h * tokenizer.w, -1
        )
        losses["legacy"] = F.binary_cross_entropy_with_logits(
            decoded["legacy_obs"], target.float()
        )
        raster_exact = ((decoded["legacy_obs"] >= 0) == target.bool()).all(-1)
        metrics["dtok/exact_raster_cell"] = raster_exact.float().mean().detach()
        metrics["dtok/exact_raster"] = raster_exact.all(-1).float().mean().detach()
    if mask is not None:
        losses["mask"] = F.binary_cross_entropy_with_logits(
            decoded["mask"], mask.float()
        )
        mask_exact = ((decoded["mask"] >= 0) == mask.bool()).all(-1)
        metrics["dtok/exact_mask_cell"] = mask_exact.float().mean().detach()
        metrics["dtok/exact_mask"] = mask_exact.all(-1).float().mean().detach()

    reduce_dims = tuple(range(code_probs.ndim - 2))
    mean_probs = code_probs.mean(dim=reduce_dims)
    batch_entropy = -(mean_probs * mean_probs.clamp_min(1e-8).log()).sum(-1).mean()
    sample_entropy = -(code_probs * code_probs.clamp_min(1e-8).log()).sum(-1).mean()
    usage = sample_entropy - batch_entropy
    reconstruction = torch.stack(list(losses.values())).sum()
    total = reconstruction + float(usage_coef) * usage

    with torch.no_grad():
        pred_state, pred_globals = tokenizer.discretize(decoded)
        true_state = state.clone()
        true_state[..., 2] = -1
        exact_cell = (pred_state == true_state).all(-1)
        exact_global = (pred_globals == globals_).all(-1)
        metrics.update(
            {
                "dtok/exact_cell": exact_cell.float().mean(),
                "dtok/exact_frame": exact_cell.all(-1).float().mean(),
                "dtok/exact_globals": exact_global.float().mean(),
                "dtok/exact_roundtrip": (exact_cell.all(-1) & exact_global)
                .float()
                .mean(),
                "dtok/code_perplexity": batch_entropy.exp(),
                "dtok/code_sample_entropy": sample_entropy,
                "dtok/oov_rate_sum": oov,
            }
        )
    metrics.update(
        {
            "dtok/reconstruction": reconstruction.detach(),
            "dtok/usage": usage.detach(),
            "dtok/total": total.detach(),
        }
    )
    return total, metrics
