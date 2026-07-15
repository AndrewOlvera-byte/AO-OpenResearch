"""Hybrid structured tokenizer for complete MicroRTS engine state."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import StructuredTokenizerConfig
from .schema import GLOBAL_WIDTH, STATE_WIDTH


def _coord_features(h: int, w: int, d: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    pos = torch.stack([x, y], dim=-1).float().reshape(h * w, 2)
    half = max(1, d // 4)
    freq = torch.exp(
        torch.arange(half).float() * (-math.log(10000.0) / max(half - 1, 1))
    )
    vals = pos[..., None] * freq
    feat = torch.cat([vals.sin(), vals.cos()], dim=-1).flatten(1)
    if feat.shape[1] < d:
        feat = F.pad(feat, (0, d - feat.shape[1]))
    return feat[:, :d]


class StructuredTokenizer(nn.Module):
    categorical = {
        "terrain": (0, 3, 0),
        "present": (1, 2, 0),
        "owner": (3, 4, 1),
        "unit_type": (4, None, 1),
        "assignment": (7, 2, 0),
        "action_type": (8, 7, 1),
        "direction": (9, 6, 1),
        "produced_type": (12, None, 1),
    }
    # Raw unit IDs are JVM-global allocation counters, not gameplay state. Keep
    # field 2 in the canonical/HDF5 schema for diagnostics, but never embed or
    # reconstruct it. Source coordinates and role bind units and issued actions.
    numeric_indices = (5, 6, 10, 11, 13, 14, 15)
    numeric_scales = (20.0, 20.0, 15.0, 15.0, 2000.0, 250.0, 250.0)
    global_scales = (2000.0, 50.0, 50.0, 50.0, 50.0, 256.0, 2.0, 1.0)

    def __init__(self, grid_hw=(16, 16), cfg: StructuredTokenizerConfig | None = None):
        super().__init__()
        self.cfg = cfg or StructuredTokenizerConfig()
        self.h, self.w = map(int, grid_hw)
        if self.cfg.downsample not in (1, 2):
            raise ValueError("v2 tokenizer supports downsample 1 (oracle) or 2")
        if self.h % self.cfg.downsample or self.w % self.cfg.downsample:
            raise ValueError("grid must be divisible by downsample")
        dc, dl = self.cfg.d_cell, self.cfg.d_latent
        self.embeds = nn.ModuleDict()
        for name, (_, size, _) in self.categorical.items():
            size = size or (self.cfg.max_unit_types + 1)
            self.embeds[name] = nn.Embedding(size, dc)
        self.numeric_proj = nn.Sequential(
            nn.Linear(len(self.numeric_indices), dc), nn.SiLU(), nn.Linear(dc, dc)
        )
        self.register_buffer(
            "coords", _coord_features(self.h, self.w, dc), persistent=False
        )
        factor = self.cfg.downsample**2
        self.compress = nn.Linear(factor * dc, dl)
        enc = nn.TransformerEncoderLayer(
            dl,
            self.cfg.n_heads,
            4 * dl,
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.spatial_encoder = nn.TransformerEncoder(
            enc, self.cfg.depth, nn.LayerNorm(dl)
        )
        self.entity_proj = nn.Linear(dc, dl)
        self.entity_pad = nn.Parameter(torch.zeros(dl))
        self.entity_type = nn.Parameter(torch.zeros(dl))
        self.game_global = nn.Linear(GLOBAL_WIDTH, dl)
        self.self_global = nn.Linear(3, dl)
        self.opp_global = nn.Linear(3, dl)
        self.global_type = nn.Parameter(torch.zeros(3, dl))
        self.n_spatial = (self.h // self.cfg.downsample) * (
            self.w // self.cfg.downsample
        )
        self.n_entity = int(self.cfg.max_entities)
        self.n_global = 3
        self.n_tokens = self.n_spatial + self.n_entity + self.n_global
        self.d_latent = dl

        self.expand = nn.Linear(dl, factor * dl)
        self.cell_context = nn.Linear(dl, dl)
        self.entity_to_cell = nn.MultiheadAttention(
            dl, self.cfg.n_heads, batch_first=True
        )
        sizes = {
            "terrain": 3,
            "present": 2,
            "owner": 4,
            "unit_type": self.cfg.max_unit_types + 1,
            "assignment": 2,
            "action_type": 7,
            "direction": 6,
            "produced_type": self.cfg.max_unit_types + 1,
        }
        self.field_heads = nn.ModuleDict(
            {k: nn.Linear(dl, n) for k, n in sizes.items()}
        )
        self.numeric_head = nn.Linear(dl, len(self.numeric_indices))
        self.global_head = nn.Sequential(
            nn.Linear(3 * dl, 2 * dl), nn.SiLU(), nn.Linear(2 * dl, GLOBAL_WIDTH)
        )
        self.legacy_head = nn.Linear(dl, self.cfg.legacy_obs_channels)
        self.mask_head = nn.Linear(dl, self.cfg.mask_width)

    def _flatten(self, state, globals_):
        lead = state.shape[:-2]
        return (
            state.reshape(-1, self.h * self.w, STATE_WIDTH),
            globals_.reshape(-1, GLOBAL_WIDTH),
            lead,
        )

    def _normalize_numeric(self, state):
        vals = state[..., list(self.numeric_indices)].float()
        scales = vals.new_tensor(self.numeric_scales)
        # Coordinates and sentinel-bearing values retain -1 as a distinct input.
        return vals / scales

    def _normalize_globals(self, globals_):
        return globals_.float() / globals_.new_tensor(self.global_scales)

    def encode(self, state: torch.Tensor, globals_: torch.Tensor) -> torch.Tensor:
        state, globals_, lead = self._flatten(state, globals_)
        x = self.numeric_proj(self._normalize_numeric(state))
        for name, (idx, size, shift) in self.categorical.items():
            n = size or (self.cfg.max_unit_types + 1)
            val = (state[..., idx] + shift).clamp(0, n - 1)
            x = x + self.embeds[name](val)
        x = x + self.coords.to(x.dtype)
        cell = x
        ds = self.cfg.downsample
        x = x.reshape(-1, self.h // ds, ds, self.w // ds, ds, self.cfg.d_cell)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            -1, self.n_spatial, ds * ds * self.cfg.d_cell
        )
        spatial = self.spatial_encoder(self.compress(x))
        # Pack occupied cells without a Python loop or GPU synchronisation.  A
        # stable sort keeps entities in spatial order and moves empty cells to
        # the end; masked slots retain the learned padding token.
        occupied = state[..., 1].bool()
        order = torch.argsort(~occupied, dim=1, stable=True)[:, : self.n_entity]
        valid = occupied.gather(1, order)
        gathered = cell.gather(1, order[..., None].expand(-1, -1, cell.shape[-1]))
        packed = self.entity_proj(gathered) + self.entity_type
        entity = torch.where(valid[..., None], packed, self.entity_pad.view(1, 1, -1))
        g = self._normalize_globals(globals_)
        gt = (
            torch.stack(
                [
                    self.game_global(g),
                    self.self_global(g[:, [1, 3, 7]]),
                    self.opp_global(g[:, [2, 4, 7]]),
                ],
                dim=1,
            )
            + self.global_type
        )
        z = torch.cat([spatial, entity, gt], dim=1)
        return z.reshape(*lead, self.n_tokens, self.d_latent)

    def decode(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        lead = z.shape[:-2]
        z = z.reshape(-1, self.n_tokens, self.d_latent)
        spatial = z[:, : self.n_spatial]
        entity = z[:, self.n_spatial : self.n_spatial + self.n_entity]
        glob = z[:, self.n_spatial + self.n_entity :]
        ds = self.cfg.downsample
        x = self.expand(spatial).reshape(
            -1, self.h // ds, self.w // ds, ds, ds, self.d_latent
        )
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.h * self.w, self.d_latent)
        x = x + self.cell_context(glob.mean(dim=1))[:, None]
        x = x + self.entity_to_cell(x, entity, entity, need_weights=False)[0]
        out = {
            name: head(x).reshape(*lead, self.h * self.w, -1)
            for name, head in self.field_heads.items()
        }
        out["numeric"] = self.numeric_head(x).reshape(*lead, self.h * self.w, -1)
        out["globals"] = self.global_head(glob.flatten(1)).reshape(*lead, GLOBAL_WIDTH)
        out["legacy_obs"] = self.legacy_head(x).reshape(*lead, self.h * self.w, -1)
        out["mask"] = self.mask_head(x).reshape(*lead, self.h * self.w, -1)
        return out

    def forward(self, state, globals_):
        z = self.encode(state, globals_)
        return self.decode(z), z

    def discretize(self, decoded: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert decoder outputs to the canonical integer schema for eval."""
        lead = decoded["numeric"].shape[:-2]
        state = torch.zeros(
            *lead,
            self.h * self.w,
            STATE_WIDTH,
            dtype=torch.long,
            device=decoded["numeric"].device,
        )
        for name, (idx, _, shift) in self.categorical.items():
            state[..., idx] = decoded[name].argmax(-1) - shift
        scales = decoded["numeric"].new_tensor(self.numeric_scales)
        nums = torch.round(decoded["numeric"] * scales).long()
        state[..., list(self.numeric_indices)] = nums
        # Generated states use a canonical ID sentinel. IDs are intentionally
        # outside the learned state and must not affect exact-state metrics.
        state[..., 2] = -1
        # Enforce empty/absent sentinels so exact-state metrics are meaningful.
        empty = state[..., 1] == 0
        for idx in (2, 3, 4, 8, 9, 10, 11, 12, 13):
            state[..., idx] = torch.where(
                empty, state[..., idx].new_full((), -1), state[..., idx]
            )
        state[..., 5:7] = torch.where(
            empty[..., None], torch.zeros_like(state[..., 5:7]), state[..., 5:7]
        )
        state[..., 7] = torch.where(
            empty, torch.zeros_like(state[..., 7]), state[..., 7]
        )
        no_assignment = state[..., 7] == 0
        for idx in (8, 9, 10, 11, 12, 13):
            state[..., idx] = torch.where(
                no_assignment, state[..., idx].new_full((), -1), state[..., idx]
            )
        state[..., 14:16] = torch.where(
            no_assignment[..., None],
            torch.zeros_like(state[..., 14:16]),
            state[..., 14:16],
        )
        gscale = decoded["globals"].new_tensor(self.global_scales)
        globals_ = torch.round(decoded["globals"] * gscale).long()
        return state, globals_


def structured_reconstruction_loss(
    tokenizer: StructuredTokenizer,
    decoded: dict,
    state: torch.Tensor,
    globals_: torch.Tensor,
    legacy_obs: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    cell_weights: torch.Tensor | None = None,
):
    """Field-aware tokenizer/grounding loss with assignment/occupancy masks."""
    losses, metrics = {}, {}
    occupied = state[..., 1].bool()
    assigned = state[..., 7].bool()
    if cell_weights is None:
        cell_weights = torch.ones_like(occupied, dtype=decoded["numeric"].dtype)
    else:
        if cell_weights.shape != occupied.shape:
            raise ValueError(
                f"cell_weights shape {tuple(cell_weights.shape)} != "
                f"state cells {tuple(occupied.shape)}"
            )
        cell_weights = cell_weights.to(decoded["numeric"].dtype)
    specs = tokenizer.categorical
    for name, (idx, _, shift) in specs.items():
        logits = decoded[name]
        target = (state[..., idx] + shift).clamp(0, logits.shape[-1] - 1).long()
        per = F.cross_entropy(
            logits.flatten(0, -2), target.flatten(), reduction="none"
        ).reshape_as(target)
        use = (
            assigned
            if name in ("action_type", "direction", "produced_type")
            else (
                occupied
                if name in ("owner", "unit_type")
                else torch.ones_like(occupied)
            )
        )
        use_f = use.to(per.dtype) * cell_weights
        losses[name] = (per * use_f).sum() / use_f.sum().clamp_min(1)
        pred = logits.argmax(-1)
        correct = (((pred == target) & use).to(use_f.dtype) * cell_weights).sum()
        metrics[f"tok/{name}_acc"] = (correct / use_f.sum().clamp_min(1)).detach()
    nt = (
        state[..., list(tokenizer.numeric_indices)].float()
        / state.new_tensor(tokenizer.numeric_scales).float()
    )
    numeric_mask = occupied[..., None].expand_as(nt).clone()
    numeric_mask[..., 2:] &= assigned[..., None]
    nerr = F.smooth_l1_loss(decoded["numeric"], nt, reduction="none")
    numeric_f = numeric_mask.to(nerr.dtype) * cell_weights[..., None]
    losses["numeric"] = (nerr * numeric_f).sum() / numeric_f.sum().clamp_min(1)
    numeric_names = (
        "hp",
        "carried",
        "target_x",
        "target_y",
        "start_tick",
        "eta",
        "remaining",
    )
    with torch.no_grad():
        raw_err = (decoded["numeric"] - nt).abs() * nt.new_tensor(
            tokenizer.numeric_scales
        )
        for i, name in enumerate(numeric_names):
            use = numeric_mask[..., i]
            use_f = use.to(raw_err.dtype) * cell_weights
            metrics[f"tok/{name}_mae"] = (
                (raw_err[..., i] * use_f).sum() / use_f.sum().clamp_min(1)
            ).detach()
    gt = globals_.float() / globals_.new_tensor(tokenizer.global_scales).float()
    losses["globals"] = F.smooth_l1_loss(decoded["globals"], gt)
    with torch.no_grad():
        ge = (decoded["globals"] - gt).abs() * gt.new_tensor(tokenizer.global_scales)
        for i, name in enumerate(
            (
                "tick",
                "self_resources",
                "opponent_resources",
                "self_reserved",
                "opponent_reserved",
                "reserved_positions",
                "winner",
                "gameover",
            )
        ):
            metrics[f"tok/global_{name}_mae"] = ge[..., i].mean().detach()
    if legacy_obs is not None:
        target = legacy_obs.movedim(-3, -1).reshape(
            *legacy_obs.shape[:-3], tokenizer.h * tokenizer.w, -1
        )
        losses["legacy"] = F.binary_cross_entropy_with_logits(
            decoded["legacy_obs"], target
        )
    if mask is not None:
        losses["mask"] = F.binary_cross_entropy_with_logits(
            decoded["mask"], mask.float()
        )
    total = sum(losses.values())
    metrics.update({f"tok/loss_{k}": v.detach() for k, v in losses.items()})
    metrics["tok/total"] = total.detach()
    return total, metrics
