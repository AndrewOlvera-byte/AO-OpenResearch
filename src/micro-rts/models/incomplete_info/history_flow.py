from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from shared.dreamerv4 import (
    BlockCausalTransformer,
    Modality,
    TokenLayout,
    add_sinusoidal_positions,
)

from .config import BeliefDynamicsConfig, HistoryConfig, JointFlowConfig


class CausalHistoryTransformer(nn.Module):
    """Factorized block-causal history with a small temporal register bank."""

    def __init__(
        self,
        n_spatial: int = 64,
        max_action_events: int = 48,
        input_dim: int = 320,
        cfg: HistoryConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or HistoryConfig()
        self.n_spatial = int(n_spatial)
        self.max_action_events = int(max_action_events)
        d = self.cfg.d_model
        self.registers = nn.Parameter(torch.zeros(self.cfg.n_registers, d))
        self.spatial_in = nn.Linear(input_dim, d)
        self.action_in = nn.Linear(input_dim, d)
        self.no_action = nn.Parameter(torch.zeros(d))
        self.boundary = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.registers, std=0.02)
        nn.init.normal_(self.no_action, std=0.02)
        nn.init.normal_(self.boundary, std=0.02)
        self.layout = TokenLayout(
            n_latents=self.cfg.n_registers,
            segments=(
                (Modality.SPATIAL, self.n_spatial),
                (Modality.ACTION, self.max_action_events),
            ),
        )
        self.transformer = BlockCausalTransformer(
            d_model=d,
            n_heads=self.cfg.n_heads,
            depth=self.cfg.depth,
            n_latents=self.cfg.n_registers,
            modality_ids=self.layout.modality_ids(),
            space_mode="encoder",
            dropout=self.cfg.dropout,
            mlp_ratio=self.cfg.mlp_ratio,
            time_every=self.cfg.time_every,
            latents_only_time=True,
        )
        self.norm = nn.LayerNorm(d)

    def forward(self, spatial, action_tokens, action_valid, is_first=None):
        if spatial.shape[1] > self.cfg.context_length:
            spatial = spatial[:, -self.cfg.context_length :]
            action_tokens = action_tokens[:, -self.cfg.context_length :]
            action_valid = action_valid[:, -self.cfg.context_length :]
            is_first = (
                is_first[:, -self.cfg.context_length :]
                if is_first is not None
                else None
            )
        b, t = spatial.shape[:2]
        registers = self.registers.view(1, 1, self.cfg.n_registers, -1).expand(
            b, t, -1, -1
        )
        action = self.action_in(action_tokens)
        action = torch.where(
            action_valid[..., None], action, self.no_action.view(1, 1, 1, -1)
        )
        tokens = torch.cat((registers, self.spatial_in(spatial), action), dim=2)
        if is_first is not None:
            tokens = tokens + is_first[..., None, None].to(tokens.dtype) * self.boundary
        tokens = add_sinusoidal_positions(tokens, True)
        output = self.norm(self.transformer(tokens))
        return {
            "registers": output[:, :, : self.cfg.n_registers],
            "spatial": output[
                :,
                :,
                self.cfg.n_registers : self.cfg.n_registers + self.n_spatial,
            ],
        }


class FutureStatePredictor(nn.Module):
    def __init__(self, history_dim=512, state_dim=320, n_state_tokens=195, n_heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(n_state_tokens, history_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.cross = nn.MultiheadAttention(history_dim, n_heads, batch_first=True)
        self.out = nn.Linear(history_dim, state_dim)

    def forward(self, registers):
        lead = registers.shape[:-2]
        memory = registers.reshape(-1, *registers.shape[-2:])
        query = self.queries[None].expand(memory.shape[0], -1, -1)
        state = query + self.cross(query, memory, memory, need_weights=False)[0]
        return self.out(state).reshape(*lead, self.queries.shape[0], -1)


class JointBeliefIntentFlow(nn.Module):
    def __init__(
        self,
        n_state_tokens=195,
        n_plan_tokens=8,
        latent_dim=320,
        history_dim=512,
        cfg: JointFlowConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or JointFlowConfig()
        self.n_state_tokens = int(n_state_tokens)
        self.n_plan_tokens = int(n_plan_tokens)
        self.latent_dim = int(latent_dim)
        d = self.cfg.d_model
        self.state_in = nn.Linear(latent_dim, d)
        self.plan_in = nn.Linear(latent_dim, d)
        self.history_in = nn.Linear(history_dim, d)
        self.state_out = nn.Linear(d, latent_dim)
        self.plan_out = nn.Linear(d, latent_dim)
        self.state_position = nn.Parameter(torch.zeros(n_state_tokens, d))
        self.plan_position = nn.Parameter(torch.zeros(n_plan_tokens, d))
        self.type_embed = nn.Parameter(torch.zeros(2, d))
        self.time_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.state_position, std=0.02)
        nn.init.normal_(self.plan_position, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d,
            self.cfg.n_heads,
            int(d * self.cfg.mlp_ratio),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, self.cfg.depth, nn.LayerNorm(d))
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)
        nn.init.zeros_(self.plan_out.weight)
        nn.init.zeros_(self.plan_out.bias)

    @property
    def n_tokens(self):
        return self.n_state_tokens + self.n_plan_tokens

    def forward(self, y_tau, tau, history_registers):
        lead = y_tau.shape[:-2]
        flat = y_tau.reshape(-1, self.n_tokens, self.latent_dim)
        state, plan = flat.split((self.n_state_tokens, self.n_plan_tokens), dim=1)
        target = torch.cat(
            (
                self.state_in(state) + self.state_position + self.type_embed[0],
                self.plan_in(plan) + self.plan_position + self.type_embed[1],
            ),
            dim=1,
        )
        flat_tau = tau.reshape(-1, 1).to(target.dtype)
        target = target + self.time_embed(flat_tau)[:, None]
        memory = self.history_in(
            history_registers.reshape(-1, *history_registers.shape[-2:])
        )
        hidden = self.decoder(target, memory)
        state_h, plan_h = hidden.split((self.n_state_tokens, self.n_plan_tokens), dim=1)
        velocity = torch.cat((self.state_out(state_h), self.plan_out(plan_h)), dim=1)
        return velocity.reshape(*lead, self.n_tokens, self.latent_dim)

    @torch.no_grad()
    def sample(self, history_registers, steps=None, noise=None):
        lead = history_registers.shape[:-2]
        steps = int(self.cfg.sample_steps if steps is None else steps)
        if steps <= 0:
            raise ValueError("flow sample steps must be positive")
        y = (
            torch.randn(
                *lead,
                self.n_tokens,
                self.latent_dim,
                device=history_registers.device,
                dtype=history_registers.dtype,
            )
            if noise is None
            else noise.clone()
        )
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                lead,
                index / steps,
                device=y.device,
                dtype=y.dtype,
            )
            y = y + dt * self(y, tau, history_registers)
        return y


class ConditionalBeliefDynamicsFlow(nn.Module):
    """State-only flow conditioned on deployable history, intent, and ego action.

    Unlike the retired joint flow, opponent plans are conditioning memory rather
    than prediction targets.  State and intent therefore cannot compete by token
    count, and sampled state hypotheses can be audited by independently
    shuffling either causal history or intent.
    """

    def __init__(
        self,
        n_state_tokens=195,
        state_dim=320,
        history_dim=512,
        plan_dim=320,
        action_dim=320,
        cfg: BeliefDynamicsConfig | None = None,
        n_spatial_tokens=0,
        spatial_hw=None,
    ):
        super().__init__()
        self.cfg = cfg or BeliefDynamicsConfig()
        self.n_state_tokens = int(n_state_tokens)
        self.state_dim = int(state_dim)
        self.n_spatial_tokens = int(n_spatial_tokens)
        self.spatial_hw = tuple(spatial_hw or ())
        d = int(self.cfg.d_model)
        self.state_in = nn.Linear(state_dim, d)
        self.history_in = nn.Linear(history_dim, d)
        self.plan_in = nn.Linear(plan_dim, d)
        self.action_in = nn.Linear(action_dim, d)
        self.state_position = nn.Parameter(torch.zeros(self.n_state_tokens, d))
        self.memory_type = nn.Parameter(torch.zeros(3, d))
        self.time_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.normal_(self.state_position, std=0.02)
        nn.init.normal_(self.memory_type, std=0.02)
        self.axial_state_position = bool(self.cfg.axial_state_position)
        if self.axial_state_position:
            if (
                len(self.spatial_hw) != 2
                or self.spatial_hw[0] * self.spatial_hw[1] != self.n_spatial_tokens
            ):
                raise ValueError(
                    "axial state positions require spatial_hw matching "
                    f"n_spatial_tokens; got {self.spatial_hw} and "
                    f"{self.n_spatial_tokens}"
                )
            self.state_row_position = nn.Parameter(torch.zeros(self.spatial_hw[0], d))
            self.state_col_position = nn.Parameter(torch.zeros(self.spatial_hw[1], d))
            nn.init.normal_(self.state_row_position, std=0.02)
            nn.init.normal_(self.state_col_position, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d,
            self.cfg.n_heads,
            int(d * self.cfg.mlp_ratio),
            self.cfg.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, self.cfg.depth, nn.LayerNorm(d))
        self.state_out = nn.Linear(d, state_dim)
        nn.init.zeros_(self.state_out.weight)
        nn.init.zeros_(self.state_out.bias)
        self.current_belief_anchor = bool(self.cfg.current_belief_anchor)
        if self.current_belief_anchor:
            self.anchor_attention = nn.MultiheadAttention(
                d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
            )
            self.anchor_norm = nn.LayerNorm(d)
            self.anchor_out = nn.Sequential(
                nn.Linear(d, int(d * self.cfg.mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(d * self.cfg.mlp_ratio), state_dim),
            )
            nn.init.zeros_(self.anchor_out[-1].weight)
            nn.init.zeros_(self.anchor_out[-1].bias)
            self.anchor_in = nn.Linear(state_dim, d)
        self.direct_action_attention = bool(self.cfg.direct_action_attention)
        if self.direct_action_attention:
            self.direct_action_attn = nn.MultiheadAttention(
                d, self.cfg.n_heads, self.cfg.dropout, batch_first=True
            )
            self.direct_action_norm = nn.LayerNorm(d)
            self.direct_action_gate = nn.Parameter(torch.tensor(-2.0))
        self.explicit_action_residual = bool(
            self.cfg.explicit_action_residual
        )
        if self.explicit_action_residual:
            self.action_residual_attn = nn.MultiheadAttention(
                d,
                self.cfg.n_heads,
                self.cfg.dropout,
                batch_first=True,
            )
            self.action_residual_norm = nn.LayerNorm(d)
            self.action_residual_mlp = nn.Sequential(
                nn.Linear(d, int(d * self.cfg.mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(d * self.cfg.mlp_ratio), state_dim),
            )
            nn.init.zeros_(self.action_residual_mlp[-1].weight)
            nn.init.zeros_(self.action_residual_mlp[-1].bias)

    def action_residual_parameters(self):
        if not self.explicit_action_residual:
            return iter(())
        modules = (
            self.action_residual_attn,
            self.action_residual_norm,
            self.action_residual_mlp,
        )
        return (parameter for module in modules for parameter in module.parameters())

    def _state_positions(self):
        position = self.state_position
        if not self.axial_state_position:
            return position
        h, w = self.spatial_hw
        axial = (
            self.state_row_position[:, None, :]
            + self.state_col_position[None, :, :]
        ).reshape(h * w, -1)
        position = position.clone()
        position[: self.n_spatial_tokens] = (
            position[: self.n_spatial_tokens] + axial
        )
        return position

    def current_anchor(self, history_registers):
        """History-derived normalized current-state latent used by v3.5."""
        if not self.current_belief_anchor:
            shape = (*history_registers.shape[:-2], self.n_state_tokens, self.state_dim)
            return history_registers.new_zeros(shape)
        lead = history_registers.shape[:-2]
        history = self.history_in(
            history_registers.reshape(-1, *history_registers.shape[-2:])
        ) + self.memory_type[0]
        query = self._state_positions()[None].expand(history.shape[0], -1, -1)
        routed = self.anchor_attention(query, history, history, need_weights=False)[0]
        anchor = self.anchor_out(self.anchor_norm(query + routed))
        return anchor.reshape(*lead, self.n_state_tokens, self.state_dim)

    def action_residual(self, y_tau, action_tokens, action_valid):
        """State-shaped, explicitly action-routed velocity correction.

        Invalid action slots are masked. Rows with no issued action receive an
        exactly zero correction while retaining a safe unmasked null key for
        ``MultiheadAttention``.
        """
        if not self.explicit_action_residual:
            return torch.zeros_like(y_tau)
        lead = y_tau.shape[:-2]
        flat = y_tau.reshape(-1, self.n_state_tokens, self.state_dim)
        actions = action_tokens.reshape(-1, *action_tokens.shape[-2:])
        valid = action_valid.reshape(-1, action_valid.shape[-1]).bool()
        any_valid = valid.any(-1)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~any_valid
        memory = self.action_in(actions) + self.memory_type[2]
        memory = torch.where(
            valid[..., None], memory, torch.zeros_like(memory)
        )
        query = self.state_in(flat)
        routed = self.action_residual_attn(
            query,
            memory,
            memory,
            key_padding_mask=~safe_valid,
            need_weights=False,
        )[0]
        correction = self.action_residual_mlp(
            self.action_residual_norm(query + routed)
        )
        correction = correction * any_valid[:, None, None].to(correction.dtype)
        return correction.reshape(*lead, self.n_state_tokens, self.state_dim)

    def forward(
        self,
        y_tau,
        tau,
        history_registers,
        plan_tokens,
        action_tokens,
        action_valid,
        current_anchor=None,
    ):
        lead = y_tau.shape[:-2]
        flat = y_tau.reshape(-1, self.n_state_tokens, self.state_dim)
        target = self.state_in(flat) + self._state_positions()
        if self.current_belief_anchor:
            anchor = (
                self.current_anchor(history_registers)
                if current_anchor is None
                else current_anchor
            )
            target = target + self.anchor_in(
                anchor.reshape(-1, self.n_state_tokens, self.state_dim)
            )
        target = target + self.time_embed(tau.reshape(-1, 1).to(target.dtype))[:, None]

        history = self.history_in(
            history_registers.reshape(-1, *history_registers.shape[-2:])
        ) + self.memory_type[0]
        plan = self.plan_in(
            plan_tokens.reshape(-1, *plan_tokens.shape[-2:])
        ) + self.memory_type[1]
        action = self.action_in(
            action_tokens.reshape(-1, *action_tokens.shape[-2:])
        ) + self.memory_type[2]
        action_mask = ~action_valid.reshape(-1, action_valid.shape[-1]).bool()
        if self.direct_action_attention:
            safe_mask = action_mask.clone()
            no_action = safe_mask.all(-1)
            safe_mask[no_action, 0] = False
            safe_action = torch.where(
                (~action_mask)[..., None], action, torch.zeros_like(action)
            )
            routed = self.direct_action_attn(
                target,
                safe_action,
                safe_action,
                key_padding_mask=safe_mask,
                need_weights=False,
            )[0]
            routed = routed * (~no_action)[:, None, None].to(routed.dtype)
            target = target + torch.sigmoid(self.direct_action_gate) * self.direct_action_norm(routed)
        memory = torch.cat((history, plan, action), dim=1)
        prefix = torch.zeros(
            action_mask.shape[0],
            history.shape[1] + plan.shape[1],
            dtype=torch.bool,
            device=action_mask.device,
        )
        hidden = self.decoder(
            target,
            memory,
            memory_key_padding_mask=torch.cat((prefix, action_mask), dim=1),
        )
        velocity = self.state_out(hidden).reshape(
            *lead, self.n_state_tokens, self.state_dim
        )
        if self.explicit_action_residual:
            velocity = velocity + self.action_residual(
                y_tau, action_tokens, action_valid
            )
        return velocity

    @torch.no_grad()
    def sample(
        self,
        history_registers,
        plan_tokens,
        action_tokens,
        action_valid,
        *,
        steps=None,
        noise=None,
    ):
        lead = history_registers.shape[:-2]
        steps = int(self.cfg.sample_steps if steps is None else steps)
        if steps <= 0:
            raise ValueError("belief dynamics sample steps must be positive")
        y = (
            torch.randn(
                *lead,
                self.n_state_tokens,
                self.state_dim,
                device=history_registers.device,
                dtype=history_registers.dtype,
            )
            if noise is None
            else noise.clone()
        )
        anchor = (
            self.current_anchor(history_registers)
            if self.current_belief_anchor
            else None
        )
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                lead, index / steps, device=y.device, dtype=y.dtype
            )
            y = y + dt * self(
                y,
                tau,
                history_registers,
                plan_tokens,
                action_tokens,
                action_valid,
                current_anchor=anchor,
            )
        return y + anchor if anchor is not None else y


@dataclass
class InferenceHistory:
    context_length: int
    spatial: list[torch.Tensor]
    action_tokens: list[torch.Tensor]
    action_valid: list[torch.Tensor]
    is_first: list[torch.Tensor]

    @classmethod
    def empty(cls, context_length: int):
        return cls(int(context_length), [], [], [], [])

    def reset(self):
        self.spatial.clear()
        self.action_tokens.clear()
        self.action_valid.clear()
        self.is_first.clear()

    def append(self, spatial, action_tokens, action_valid, is_first):
        if bool(is_first.any()):
            self.reset()
        for store, value in (
            (self.spatial, spatial),
            (self.action_tokens, action_tokens),
            (self.action_valid, action_valid),
            (self.is_first, is_first),
        ):
            store.append(value)
            del store[: max(0, len(store) - self.context_length)]

    def tensors(self):
        if not self.spatial:
            raise RuntimeError("inference history is empty")
        return tuple(
            torch.stack(values, dim=1)
            for values in (
                self.spatial,
                self.action_tokens,
                self.action_valid,
                self.is_first,
            )
        )
