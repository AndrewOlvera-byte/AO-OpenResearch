"""``AtariWorldModel`` — DreamerV4 dynamics over Atari latent tokens.

Identical block-causal transformer core to the MicroRTS build (the vendored
``BlockCausalTransformer`` primitive), differing only in the **action encoder**:
Atari actions are a single ``Discrete(num_actions)``, so a plain ``Embedding`` maps
the action index to one action token per frame (vs the MicroRTS per-cell GridNet
encoder). Each timestep is tokenized as ``[action, spatial_latents..., registers...]``;
the transformer context feeds a deterministic next-latent head (imagination + latent
MSE) and a shortcut-forcing velocity head (Dreamer 4 generative dynamics), plus the
reward and continue heads.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from shared.dreamerv4 import (
    BlockCausalTransformer,
    Modality,
    TokenLayout,
    TwoHot,
    add_sinusoidal_positions,
)

from .config import DynamicsConfig


def _time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / max(half, 1))
    ang = t.unsqueeze(-1).float() * freqs
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if emb.shape[-1] < dim:
        emb = torch.cat([emb, emb[..., :1] * 0], dim=-1)
    return emb


class AtariWorldModel(nn.Module):
    def __init__(self, n_spatial: int, d_latent: int, num_actions: int, cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_spatial = n_spatial
        self.d_latent = d_latent
        d = cfg.d_model

        self.latent_in = nn.Linear(d_latent, d)
        self.action_embed = nn.Embedding(num_actions, d)
        self.registers = nn.Parameter(torch.randn(cfg.n_register, d) * 0.02)

        segments = [(Modality.ACTION, 1), (Modality.SPATIAL, n_spatial),
                    (Modality.REGISTER, cfg.n_register)]
        self.layout = TokenLayout(n_latents=0, segments=tuple(segments))
        sl = self.layout.slices()
        self.spatial_slice = sl[Modality.SPATIAL]
        self.register_slice = sl[Modality.REGISTER]

        self.transformer = BlockCausalTransformer(
            d_model=d, n_heads=cfg.n_heads, depth=cfg.depth, n_latents=0,
            modality_ids=self.layout.modality_ids(), space_mode=cfg.space_mode,
            dropout=cfg.dropout, mlp_ratio=cfg.mlp_ratio,
            time_every=cfg.time_every, latents_only_time=False,
        )

        self.next_head = nn.Linear(d, d_latent)
        nn.init.zeros_(self.next_head.weight)
        nn.init.zeros_(self.next_head.bias)
        # Symlog two-hot reward head: emits `reward_bins` logits, zero-init so the
        # initial expected reward is symexp(0)=0 (a stable start on sparse ±1 rewards).
        self.reward_head = nn.Linear(d, cfg.reward_bins)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        self.reward_coder = TwoHot(cfg.reward_bins)
        self.continue_head = nn.Linear(d, 1)

        self.flow_in = nn.Linear(d_latent, d)
        self.flow_cond = nn.Linear(d, d)
        self.flow_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d_latent))

    def contextualize(self, z_seq: torch.Tensor, act_seq: torch.Tensor) -> dict:
        """z_seq (B,T,n_spatial,d_latent), act_seq (B,T) long -> context + scalars."""
        B, T = z_seq.shape[:2]
        d = self.cfg.d_model
        spatial = self.latent_in(z_seq)
        action = self.action_embed(act_seq.long()).unsqueeze(2)   # (B,T,1,d)
        reg = self.registers.view(1, 1, -1, d).expand(B, T, -1, -1)
        tokens = torch.cat([action, spatial, reg], dim=2)
        tokens = add_sinusoidal_positions(tokens, self.cfg.scale_pos_embeds)
        x = self.transformer(tokens)
        h = x[:, :, self.spatial_slice, :]
        reg_out = x[:, :, self.register_slice, :].mean(dim=2)
        reward_logits = self.reward_head(reg_out)                  # (B,T,reward_bins)
        return {
            "h": h,
            "reward_logits": reward_logits,
            "reward": self.reward_coder.mean(reward_logits),       # (B,T) decoded scalar
            "continue_logit": self.continue_head(reg_out).squeeze(-1),
        }

    def next_latent(self, h: torch.Tensor, z_current: torch.Tensor | None = None) -> torch.Tensor:
        delta = self.next_head(h)
        return delta if z_current is None else z_current + delta

    def flow_velocity(self, h, z_tau, tau, d):
        cond = _time_embed(tau, self.cfg.d_model) + _time_embed(d, self.cfg.d_model)
        cond = self.flow_cond(cond)[:, :, None, :]
        return self.flow_head(h + self.flow_in(z_tau) + cond)

    def flow_sample(self, h: torch.Tensor, steps: int) -> torch.Tensor:
        """Integrate the shortcut-forcing velocity field noise -> next latent.

        Euler-integrates ``dz/dtau`` from ``tau=0`` (noise) to ``tau=1`` in ``steps``
        equal shortcut steps of size ``d=1/steps``; ``d`` matches the dyadic step
        sizes the flow was trained on when ``steps`` is a power of two. ``h`` is the
        block-causal context (B,T,n_spatial,d); returns (B,T,n_spatial,d_latent)."""
        B, T = h.shape[:2]
        z = torch.randn(B, T, self.n_spatial, self.d_latent, device=h.device, dtype=h.dtype)
        dt = 1.0 / steps
        d = torch.full((B, T), dt, device=h.device)
        for i in range(steps):
            tau = torch.full((B, T), i * dt, device=h.device)
            z = z + dt * self.flow_velocity(h, z, tau, d)
        return z

    def predict(self, z_seq: torch.Tensor, act_seq: torch.Tensor) -> dict:
        ctx = self.contextualize(z_seq, act_seq)
        ctx["next_latent"] = self.next_latent(ctx["h"], z_seq)
        return ctx
