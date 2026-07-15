"""``WorldModel`` — Dreamer 4 shortcut-forcing dynamics over MicroRTS latent tokens.

Faithful to the Dreamer 4 recipe (verified against nicklashansen/dreamer4): the
block-causal transformer *is* the denoiser. Every frame's spatial tokens enter the
transformer as a **noised** interpolation ``z_tilde = (1-sigma)*noise + sigma*z``
with a per-frame noise level (diffusion forcing), conditioned by two extra tokens:

- ``SHORTCUT_SIGNAL`` — the discretized signal level ``sigma`` (0 = pure noise,
  ``k_max-1``-th bin ≈ clean; index ``k_max-1`` is what context frames get at
  rollout, matching upstream),
- ``SHORTCUT_STEP`` — the dyadic shortcut step size ``d = 2^-step_idx``.

The model **x-predicts**: a zero-init linear head reads the spatial token outputs
and emits ``x1_hat``, the *clean* latents of the same frame. Temporal causality
means frame ``t`` is denoised from its own noise + the (noised) past, which at
rollout time turns into autoregressive generation: append pure noise for frame
``t+1``, Euler-integrate ``z <- z + d * (x1_hat - z)/(1 - tau)``.

Per-frame token layout (vendored ``TokenLayout`` / ``Modality``):
``[action x n_action, signal, step, spatial x n_spatial, register x n_register]``
with full within-frame mixing (``wm_agent``) and causal time attention.

The **action encoder** is CNN-GridNet style and produces *multiple* action
embeddings: the 7 per-cell components are embedded and summed on the ``H x W``
grid (like the obs planes), then the canonical GridNet stride-2 conv stack
downsamples to the same ``H/4 x W/4`` grid as the tokenizer latents — one action
token per bottleneck cell, spatially aligned with the spatial latent tokens.

Action/frame alignment: the action token at frame slot ``t`` is the action chosen
at frame ``t-1`` (the action that *produced* frame ``t``). ``is_first`` marks
post-reset frames: their action signal is replaced by a learned embedding, since
the previous action belongs to a different episode. The reward/continue heads read
the register tokens of the **clean pass** and are arrive-aligned the same way:
the readout at slot ``t`` predicts the reward/continue of the ``t-1 -> t``
transition.

**Joint-action conditioning (v3, NEXT_PLAN.md):** MicroRTS is simultaneous-move —
the true dynamics are ``z' = f(z, a_self, a_opp)``. Conditioning on ``a_self``
alone leaves the opponent an unmodeled latent confounder, and few-step flow
sampling averages the multimodal opponent response into near-static latents (the
v2 failure). So the action encoder embeds BOTH players' per-cell actions on the
same H x W grid and sums them before the conv trunk. The two channels use
*separate* embedding tables — a plain shared-table sum would be order-invariant
(emb(a)+emb(b) == emb(b)+emb(a)), erasing who did what; separate tables are the
role embedding, fused multiplicatively into each table. When the opponent's
action is unavailable — imagination without an opponent policy, or the
train-time **opponent dropout** that keeps the marginal model usable (and is the
forward-compatibility hook for fog of war) — a learned ``unknown_opp`` embedding
replaces that channel per frame. Both channels share the ``is_first`` masking
and the shift-by-one alignment; the action token count is unchanged.
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


def _group_norm(channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


# Per-cell GridNet action components (action_nvec[1:]): which conditional
# component is live for each action TYPE (component 0). Type indices: 0 NOOP,
# 1 move, 2 harvest, 3 return, 4 produce, 5 attack. Component i>0 is only read
# by the engine when the type matches; everywhere else the stored/executed
# value is 0 and a sampled value would be junk.
_COMPONENT_ACTIVE_TYPE = (None, 1, 2, 3, 4, 4, 5)  # comp -> required type


def zero_inactive_components(action: torch.Tensor) -> torch.Tensor:
    """Zero every conditional component not selected by its cell's action type.

    ``action`` (..., n_comp) long. Keeps sampled per-cell actions on the same
    support as engine-executed ones (unused components are stored as 0), so a
    head-sampled opponent action matches the distribution the encoder trained on.
    """
    atype = action[..., 0]
    keep = [torch.ones_like(atype, dtype=torch.bool)]
    keep += [atype == t for t in _COMPONENT_ACTIVE_TYPE[1:]]
    return action * torch.stack(keep, dim=-1).to(action.dtype)


# MicroRTS 27-channel obs layout (see loss.dreamer._OBS_GROUPS): owner group is
# channels [10:13) (none/own/enemy), current-action group [21:27) (21 = idle).
_OWNER_ENEMY_CH = 12
_ACTION_NONE_CH = 21
_MRTS_OBS_CHANNELS = 27


def opponent_source_cells(obs: torch.Tensor) -> torch.Tensor:
    """(B,[T],H*W) bool: cells where the OPPONENT can issue an action this frame
    (an idle enemy unit sits there) — the opponent-side twin of
    ``loss.dreamer.mask_actions_to_sources``'s legality mask. ``obs`` may be a
    decoded reconstruction (floats); thresholded at 0.5 like elsewhere."""
    src = (obs[..., _OWNER_ENEMY_CH, :, :] > 0.5) & \
          (obs[..., _ACTION_NONE_CH, :, :] > 0.5)
    return src.flatten(-2)


class OpponentPolicyHead(nn.Module):
    """Trunk spatial features -> per-cell opponent action logits (v4.4 BC head).

    Mirrors the tokenizer's mask decoder: project the ``n_spatial`` trunk
    tokens back onto the ``H/4 x W/4`` grid, deconv up to ``H x W``, and emit
    the concatenated per-component logits (``sum(comp_sizes)`` per cell).
    Reading the TRUNK (not the raw latent) is the point: the BC gradient flows
    through the dynamics transformer, forcing its hidden state to track
    opponent intent — the auxiliary-conditioning signal — while doubling as
    the sampler imagination uses to play an explicit opponent.
    """

    def __init__(self, d_model: int, comp_sizes, grid_hw: tuple[int, int],
                 channels: int = 64) -> None:
        super().__init__()
        self.comp_sizes = [int(x) for x in comp_sizes]
        self.h, self.w = int(grid_hw[0]), int(grid_hw[1])
        self.hb, self.wb = self.h // 4, self.w // 4
        total = sum(self.comp_sizes)
        ch = int(channels)
        self.from_trunk = nn.Conv2d(d_model, ch, 1)
        self.decoder = nn.Sequential(
            _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), _group_norm(ch), nn.ReLU(),
            nn.Conv2d(ch, total, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``feats`` (B,[T],n_spatial,d_model) -> logits (B,[T],H*W,sum(comp_sizes))."""
        lead = feats.shape[:-2]
        x = feats.reshape(-1, self.hb, self.wb, feats.shape[-1]).permute(0, 3, 1, 2)
        logits = self.decoder(self.from_trunk(x))               # (N, total, H, W)
        logits = logits.flatten(2).transpose(1, 2)              # (N, H*W, total)
        return logits.reshape(*lead, self.h * self.w, logits.shape[-1])


class GridActionEncoder(nn.Module):
    """Per-cell GridNet action ``(...,H*W,7)`` -> ``n_action`` action tokens.

    Mirrors how the tokenizer treats the obs: embed per-cell content on the full
    grid, then downsample with the GridNet conv trunk (GroupNorm + ReLU, two
    stride-2 stages) to the ``H/4 x W/4`` bottleneck — so action token ``i`` is
    spatially aligned with spatial latent token ``i``.
    """

    def __init__(self, cell_nvec, grid_hw: tuple[int, int], d_model: int,
                 emb_dim: int = 32, channels: int = 64) -> None:
        super().__init__()
        self.comp_sizes = [int(x) for x in cell_nvec]
        self.n_comp = len(self.comp_sizes)
        self.h, self.w = int(grid_hw[0]), int(grid_hw[1])
        self.emb_dim = int(emb_dim)
        self.n_action = (self.h // 4) * (self.w // 4)

        self.embeds = nn.ModuleList(nn.Embedding(sz, emb_dim) for sz in self.comp_sizes)
        # Opponent channel: separate tables = the player-role distinction (see
        # module docstring); a shared-table sum would be order-invariant.
        self.opp_embeds = nn.ModuleList(nn.Embedding(sz, emb_dim) for sz in self.comp_sizes)
        # Replaces the opponent channel when the opponent's action is unknown
        # (imagination without an opponent policy; train-time opponent dropout).
        self.unknown_opp = nn.Parameter(torch.zeros(emb_dim))
        # Replaces the per-cell action signal on frames whose previous action is
        # unknown/invalid (episode starts, first frame of a sampled window).
        self.first_embed = nn.Parameter(torch.zeros(emb_dim))

        self.encoder = nn.Sequential(
            nn.Conv2d(emb_dim, 32, 3, padding=1), _group_norm(32), nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1, stride=2), _group_norm(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, stride=2), _group_norm(channels), nn.ReLU(),
        )
        self.to_tokens = nn.Conv2d(channels, d_model, 1)

    def _embed(self, action: torch.Tensor, tables: nn.ModuleList) -> torch.Tensor:
        emb = 0.0
        for i, table in enumerate(tables):
            emb = emb + table(action[..., i].clamp_min(0))        # (BT, H*W, emb)
        return emb

    def forward(self, action: torch.Tensor, no_action: torch.Tensor | None = None,
                opp_action: torch.Tensor | None = None,
                opp_unknown: torch.Tensor | None = None,
                return_grid: bool = False):
        """Joint-action tokens.

        ``action``/``opp_action`` (B,T,H*W,n_comp) long; ``no_action`` (B,T) bool
        replaces the whole per-cell signal with ``first_embed``; ``opp_unknown``
        (B,T) bool replaces the opponent channel with ``unknown_opp`` (also the
        fallback when ``opp_action`` is None). Returns (B,T,n_action,d); with
        ``return_grid`` also the pre-token conv features (B,T,n_action,channels),
        spatially aligned 1:1 with the latent grid — the positional-injection
        path (``action_inject: add``) adds a projection of these to the matching
        spatial latent tokens, giving each cell a direct (non-attention) route
        from its own action to its own next state."""
        B, T = action.shape[:2]
        flat = action.reshape(B * T, self.h * self.w, self.n_comp).long()
        emb = self._embed(flat, self.embeds)
        unk = self.unknown_opp.view(1, 1, -1)
        if opp_action is None:
            emb = emb + unk
        else:
            oflat = opp_action.reshape(B * T, self.h * self.w, self.n_comp).long()
            oemb = self._embed(oflat, self.opp_embeds)
            if opp_unknown is not None:
                u = opp_unknown.reshape(B * T, 1, 1).to(oemb.dtype)
                oemb = oemb * (1.0 - u) + unk * u
            emb = emb + oemb
        if no_action is not None:
            f = no_action.reshape(B * T, 1, 1).to(emb.dtype)
            emb = emb * (1.0 - f) + self.first_embed.view(1, 1, -1) * f
        m = emb.reshape(B * T, self.h, self.w, self.emb_dim).permute(0, 3, 1, 2)
        feat = self.encoder(m)                                    # (BT, ch, h/4, w/4)
        tok = self.to_tokens(feat)                                # (BT, d, h/4, w/4)
        tok = tok.flatten(2).transpose(1, 2)                      # (BT, n_action, d)
        tok = tok.reshape(B, T, self.n_action, -1)
        if return_grid:
            grid = feat.flatten(2).transpose(1, 2).reshape(B, T, self.n_action, -1)
            return tok, grid
        return tok


class WorldModel(nn.Module):
    def __init__(self, n_spatial: int, d_latent: int, cell_nvec,
                 grid_hw: tuple[int, int], cfg: DynamicsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_spatial = n_spatial
        self.d_latent = d_latent
        d = cfg.d_model

        assert cfg.k_max >= 2 and (cfg.k_max & (cfg.k_max - 1)) == 0, \
            "k_max must be a power of two (dyadic shortcut steps)"
        self.k_max = int(cfg.k_max)
        self.emax = int(math.log2(self.k_max))

        self.latent_in = nn.Linear(d_latent, d)
        self.action_encoder = GridActionEncoder(
            cell_nvec, grid_hw, d, emb_dim=cfg.action_emb, channels=cfg.action_channels,
        )
        self.n_action = self.action_encoder.n_action

        # Positional action injection (action_inject: "add"): per-cell action
        # features added straight onto the matching spatial latent tokens. The
        # action-token attention path stays; this is an ADDITIONAL direct route.
        # Zero-init: at init the model computes exactly what "tokens" mode does,
        # and the projection opens only as far as the gradient pulls it.
        inject = str(getattr(cfg, "action_inject", "tokens"))
        assert inject in ("tokens", "add"), f"unknown action_inject {inject!r}"
        self.action_inject = inject == "add"
        if self.action_inject:
            assert self.n_action == n_spatial, \
                "positional injection needs the action grid aligned with the latent grid"
            self.action_inject_proj = nn.Linear(cfg.action_channels, d)
            nn.init.zeros_(self.action_inject_proj.weight)
            nn.init.zeros_(self.action_inject_proj.bias)
        # v4.4 opponent-policy head (None when disabled; see OpponentPolicyHead).
        self.opp_head = OpponentPolicyHead(
            d, cell_nvec, grid_hw, channels=getattr(cfg, "opp_head_channels", 64),
        ) if getattr(cfg, "opp_head", False) else None
        self.registers = nn.Parameter(torch.randn(cfg.n_register, d) * 0.02)

        # Shortcut conditioning (sizes match the vendored Dynamics primitive).
        self.signal_embed = nn.Embedding(self.k_max + 1, d)
        self.step_embed = nn.Embedding(self.emax + 1, d)

        segments = [
            (Modality.ACTION, self.n_action),
            (Modality.SHORTCUT_SIGNAL, 1),
            (Modality.SHORTCUT_STEP, 1),
            (Modality.SPATIAL, n_spatial),
            (Modality.REGISTER, cfg.n_register),
        ]
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

        # x-prediction head (zero-init, like upstream flow_x_head): predicts the
        # CLEAN latents of each frame from the transformer's spatial outputs.
        self.flow_x_head = nn.Linear(d, d_latent)
        nn.init.zeros_(self.flow_x_head.weight)
        nn.init.zeros_(self.flow_x_head.bias)

        # Scalar predictors read the pooled register tokens of the clean pass.
        # Symlog two-hot reward (DreamerV3 stabilizer); continue is a Bernoulli logit.
        self.reward_head = nn.Linear(d, cfg.reward_bins)
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)
        self.reward_coder = TwoHot(cfg.reward_bins)
        self.continue_head = nn.Linear(d, 1)

        # RMS of the (frozen) tokenizer's latents over the training data. The flow
        # objective interpolates against unit-variance noise, so the world model
        # operates on latents divided by this scale; ``sample_next`` multiplies it
        # back. Set from data via ``set_latent_scale`` (tokenizer phase measures it).
        self.register_buffer("latent_scale", torch.ones(()))

    # --- latent normalization ----------------------------------------------
    def set_latent_scale(self, scale: float) -> None:
        self.latent_scale.fill_(max(float(scale), 1e-6))

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        """Raw tokenizer latents -> the unit-RMS space the denoiser works in."""
        return z / self.latent_scale

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.latent_scale

    # --- alignment helpers ------------------------------------------------
    def shift_actions(self, action: torch.Tensor,
                      is_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """``action[t]`` = chosen AT frame t -> slot action = action that PRODUCED frame t.

        Returns (prev_action, no_action) where ``no_action`` flags slots whose
        previous action is unknown: the window's first slot, and episode starts.
        """
        B, T = action.shape[:2]
        prev = torch.zeros_like(action)
        if T > 1:
            prev[:, 1:] = action[:, :-1]
        no_action = torch.zeros(B, T, dtype=torch.bool, device=action.device)
        no_action[:, 0] = True
        if is_first is not None:
            no_action |= is_first.bool().to(action.device)
        return prev, no_action

    @staticmethod
    def _shift_frames(x: torch.Tensor) -> torch.Tensor:
        """Frame-aligned -> slot-aligned (slot t carries frame t-1's value)."""
        prev = torch.zeros_like(x)
        if x.shape[1] > 1:
            prev[:, 1:] = x[:, :-1]
        return prev

    # --- core pass ---------------------------------------------------------
    def _forward(self, z_in: torch.Tensor, action: torch.Tensor,
                 is_first: torch.Tensor | None,
                 signal_idx: torch.Tensor, step_idx: torch.Tensor,
                 opponent_action: torch.Tensor | None = None,
                 opp_unknown: torch.Tensor | None = None) -> torch.Tensor:
        """One transformer pass. ``z_in`` (B,T,n_spatial,d_latent) is CLEAN or NOISED
        spatial input; ``signal_idx``/``step_idx`` (B,T) condition each frame.
        ``opponent_action`` (B,T,H*W,n_comp) is frame-aligned like ``action`` and
        shifted the same way; ``opp_unknown`` (B,T) bool marks frames whose
        opponent action should be treated as unknown (dropout / imagination)."""
        B, T = z_in.shape[:2]
        d = self.cfg.d_model
        prev_act, no_act = self.shift_actions(action, is_first)
        prev_opp = self._shift_frames(opponent_action.long()) \
            if opponent_action is not None else None
        prev_unk = self._shift_frames(opp_unknown.bool()) \
            if opp_unknown is not None else None
        if self.action_inject:
            act_tok, act_grid = self.action_encoder(
                prev_act, no_act, prev_opp, prev_unk, return_grid=True)
        else:
            act_tok = self.action_encoder(prev_act, no_act, prev_opp, prev_unk)
        sig_tok = self.signal_embed(signal_idx.long())[:, :, None, :]
        stp_tok = self.step_embed(step_idx.long())[:, :, None, :]
        spatial = self.latent_in(z_in)                             # (B,T,n_spatial,d)
        if self.action_inject:
            spatial = spatial + self.action_inject_proj(act_grid)  # per-cell route
        reg = self.registers.view(1, 1, -1, d).expand(B, T, -1, -1)
        tokens = torch.cat([act_tok, sig_tok, stp_tok, spatial, reg], dim=2)
        tokens = add_sinusoidal_positions(tokens, self.cfg.scale_pos_embeds)
        return self.transformer(tokens)

    def _clean_idxs(self, B: int, T: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioning for fully-denoised frames (context at rollout): the highest
        *trained* signal bin ``k_max-1`` and the finest step ``emax`` (upstream)."""
        sig = torch.full((B, T), self.k_max - 1, dtype=torch.long, device=device)
        stp = torch.full((B, T), self.emax, dtype=torch.long, device=device)
        return sig, stp

    # --- public API ---------------------------------------------------------
    def denoise(self, z_tilde: torch.Tensor, action: torch.Tensor,
                is_first: torch.Tensor | None,
                signal_idx: torch.Tensor, step_idx: torch.Tensor,
                return_registers: bool = False,
                return_spatial: bool = False,
                opponent_action: torch.Tensor | None = None,
                opp_unknown: torch.Tensor | None = None):
        """Noised pass -> ``x1_hat`` (B,T,n_spatial,d_latent), the predicted clean
        latents of each frame (x-prediction, shortcut forcing objective).

        Operates in **normalized** latent space: ``z_tilde`` mixes unit-variance
        noise with ``normalize``-d latents, and ``x1_hat`` is normalized too.
        With ``return_registers`` also returns the pooled register outputs
        (B,T,d_model) so the reward/continue heads can read the same pass; with
        ``return_spatial`` also the raw spatial trunk outputs (B,T,n_spatial,
        d_model) — the opponent-policy head's input. Tuple order:
        ``(x1_hat[, registers][, spatial])``."""
        x = self._forward(z_tilde, action, is_first, signal_idx, step_idx,
                          opponent_action, opp_unknown)
        spatial = x[:, :, self.spatial_slice, :]
        x1_hat = self.flow_x_head(spatial)
        out = (x1_hat,)
        if return_registers:
            out += (x[:, :, self.register_slice, :].mean(dim=2),)
        if return_spatial:
            out += (spatial,)
        return out if len(out) > 1 else x1_hat

    def contextualize(self, z: torch.Tensor, action: torch.Tensor,
                      is_first: torch.Tensor | None = None,
                      opponent_action: torch.Tensor | None = None,
                      opp_unknown: torch.Tensor | None = None) -> dict:
        """Clean pass over encoded latents -> context + arrive-aligned scalars.

        ``reward``/``continue_logit`` at slot ``t`` describe the ``t-1 -> t``
        transition (slot 0 has no valid target — callers must skip it). ``z`` is
        in RAW tokenizer space; normalization happens here."""
        B, T = z.shape[:2]
        z = self.normalize(z)
        sig, stp = self._clean_idxs(B, T, z.device)
        x = self._forward(z, action, is_first, sig, stp, opponent_action, opp_unknown)
        h = x[:, :, self.spatial_slice, :]
        reg_out = x[:, :, self.register_slice, :].mean(dim=2)      # (B,T,d)
        reward_logits = self.reward_head(reg_out)
        return {
            "h": h,
            "reward_logits": reward_logits,
            "reward": self.reward_coder.mean(reward_logits),       # (B,T) decoded scalar
            "continue_logit": self.continue_head(reg_out).squeeze(-1),
        }

    @torch.no_grad()
    def sample_opponent(self, feats: torch.Tensor,
                        source_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Sample a per-cell opponent action from the opponent-policy head.

        ``feats`` (B,n_spatial,d_model) — trunk spatial outputs of a clean pass
        at the frame the opponent acts in (e.g. ``contextualize(...)["h"][:, -1]``);
        ``source_mask`` (B,H*W) bool marks cells where the opponent can act
        (see ``opponent_source_cells`` on a decoded frame) — actions elsewhere
        are zeroed, since the head is only trained at source cells. Conditional
        components not selected by the sampled type are zeroed too, matching
        the engine-executed support the action encoder trained on.
        Returns (B,H*W,n_comp) long."""
        assert self.opp_head is not None, "world model built without opp_head"
        logits = self.opp_head(feats)                          # (B,H*W,total)
        comps, off = [], 0
        for n in self.opp_head.comp_sizes:
            dist = torch.distributions.Categorical(logits=logits[..., off:off + n].float())
            comps.append(dist.sample())
            off += n
        action = torch.stack(comps, dim=-1).long()             # (B,H*W,n_comp)
        action = zero_inactive_components(action)
        if source_mask is not None:
            action = action * source_mask.to(action.dtype).unsqueeze(-1)
        return action

    @torch.no_grad()
    def sample_next(self, z_ctx: torch.Tensor, action: torch.Tensor,
                    is_first: torch.Tensor | None, steps: int,
                    opponent_action: torch.Tensor | None = None,
                    opp_unknown: torch.Tensor | None = None) -> torch.Tensor:
        """Generate the next frame's latents by shortcut denoising.

        ``z_ctx`` (B,Tc,n_spatial,d_latent) clean context; ``action`` (B,Tc,...)
        actions chosen at the context frames (``action[:, -1]`` is the action that
        produces the new frame); ``opponent_action``/``opp_unknown`` follow the
        same frame alignment (None = opponent unknown everywhere). Euler-integrates
        ``steps`` shortcut steps (power of two <= k_max):
        ``z <- z + d * (x1_hat - z)/(1 - tau)``.
        ``z_ctx`` is RAW tokenizer latents; integration runs in normalized space
        and the returned (B,n_spatial,d_latent) frame is denormalized back.
        """
        z_ctx = self.normalize(z_ctx)
        B, Tc = z_ctx.shape[:2]
        device = z_ctx.device
        K = int(steps)
        assert K >= 1 and (K & (K - 1)) == 0 and K <= self.k_max, \
            "flow steps must be a power of two <= k_max"
        e = int(math.log2(K))

        z_cur = torch.randn(B, 1, self.n_spatial, self.d_latent, device=device,
                            dtype=z_ctx.dtype)
        # Slot-aligned inputs for [context..., new frame]. The new frame's slot
        # action comes from action[:, -1] via the shift; its own column is dummy.
        act_pad = torch.cat([action, torch.zeros_like(action[:, :1])], dim=1)
        opp_pad = torch.cat([opponent_action, torch.zeros_like(opponent_action[:, :1])],
                            dim=1) if opponent_action is not None else None
        unk_pad = torch.cat([opp_unknown.bool(),
                             torch.zeros(B, 1, dtype=torch.bool, device=device)],
                            dim=1) if opp_unknown is not None else None
        if is_first is not None:
            first_pad = torch.cat(
                [is_first.bool(), torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
        else:
            first_pad = None

        sig, stp = self._clean_idxs(B, Tc + 1, device)
        stp[:, -1] = e
        dt = 1.0 / K
        for i in range(K):
            tau = i * dt
            sig[:, -1] = (i * self.k_max) // K                     # sigma grid index
            z_in = torch.cat([z_ctx, z_cur], dim=1)
            x1_hat = self.denoise(z_in, act_pad, first_pad, sig, stp,
                                  opponent_action=opp_pad, opp_unknown=unk_pad)[:, -1:]
            b = (x1_hat - z_cur) / max(1.0 - tau, 1e-4)
            z_cur = z_cur + dt * b
        return self.denormalize(z_cur[:, 0])
