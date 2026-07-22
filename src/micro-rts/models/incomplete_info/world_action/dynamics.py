from __future__ import annotations

import torch
import torch.nn as nn

from core.registry import register

from .attention import CrossBlock, TransformerBlock
from .config import FactorizedDynamicsConfig
from .encoder import BRANCHES, split_branches


class ConditionalBranchFlow(nn.Module):
    """Flow velocity for one causal branch with an explicit condition route."""

    def __init__(self, n_tokens, condition_dim, cfg):
        super().__init__()
        d = cfg.d_model
        self.n_tokens = int(n_tokens)
        self.in_proj = nn.Linear(cfg.d_latent, d)
        self.condition_in = nn.Linear(condition_dim, d)
        self.time = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        self.position = nn.Parameter(torch.zeros(n_tokens, d))
        nn.init.normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            CrossBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth)
        )
        self.mixers = nn.ModuleList(
            TransformerBlock(d, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(max(1, cfg.depth // 2))
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.d_latent)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noisy, tau, condition):
        lead = noisy.shape[:-2]
        value = self.in_proj(noisy) + self.position
        value = value + self.time(tau[..., None].to(value.dtype))[..., None, :]
        value = value.reshape(-1, self.n_tokens, value.shape[-1])
        memory = self.condition_in(condition).reshape(-1, condition.shape[-2], value.shape[-1])
        for index, cross in enumerate(self.blocks):
            value = self.mixers[index % len(self.mixers)](cross(value, memory))
        return self.out(self.norm(value)).reshape(*lead, self.n_tokens, -1)


class FactorizedWorldActionDynamics(nn.Module):
    """Intrinsic/extrinsic/interaction flow with immutable static state.

    The opponent branch has no direct path from the simultaneous ego action.
    The interaction branch is the sole joint-action route.
    """

    def __init__(self, action_dim=320, opponent_plan_dim=320, cfg=None):
        super().__init__()
        self.cfg = cfg or FactorizedDynamicsConfig()
        sizes = self.cfg.branch_sizes
        dlat = self.cfg.d_latent
        self.action_in = nn.Linear(action_dim, dlat)
        self.plan_in = nn.Linear(opponent_plan_dim, dlat)
        self.self_flow = ConditionalBranchFlow(
            sizes[0], dlat, self.cfg
        )
        self.opponent_flow = ConditionalBranchFlow(
            sizes[1], dlat, self.cfg
        )
        self.interaction_flow = ConditionalBranchFlow(
            sizes[3], dlat, self.cfg
        )

    def velocity(self, noisy, tau, departure, ego_action, opponent_plan):
        source = split_branches(departure, self.cfg.branch_sizes)
        target = split_branches(noisy, self.cfg.branch_sizes)
        action = self.action_in(ego_action.mean(-2))[..., None, :]
        plan = self.plan_in(opponent_plan)
        static = source["static"]
        intrinsic_condition = torch.cat((source["self"], static, action), dim=-2)
        extrinsic_condition = torch.cat((source["opponent"], static, plan), dim=-2)
        interaction_condition = torch.cat(
            (source["self"], source["opponent"], static, action, plan), dim=-2
        )
        pieces = {
            "self": self.self_flow(target["self"], tau, intrinsic_condition),
            "opponent": self.opponent_flow(
                target["opponent"], tau, extrinsic_condition
            ),
            "static": torch.zeros_like(target["static"]),
            "interaction": self.interaction_flow(
                target["interaction"], tau, interaction_condition
            ),
        }
        return torch.cat(tuple(pieces[name] for name in BRANCHES), dim=-2)

    def transition(self, departure, ego_action, opponent_plan, steps=None, noise=None):
        """Integrate the flow to the predicted next belief (gradient-enabled).

        Euler-samples the velocity field from ``noise`` (default N(0,1)) to a
        branch delta, adds it to ``departure``, and copies the static branch
        through exactly once.  ``sample`` is the ``no_grad`` wrapper used at
        inference; training losses (multi-horizon rollout, counterfactuals) call
        ``transition`` directly so gradients flow through the rollout.
        """
        steps = int(steps or self.cfg.sample_steps)
        value = torch.randn_like(departure) if noise is None else noise
        dt = 1.0 / steps
        for index in range(steps):
            tau = torch.full(
                departure.shape[:-2],
                index / steps,
                device=departure.device,
                dtype=departure.dtype,
            )
            value = value + dt * self.velocity(
                value, tau, departure, ego_action, opponent_plan
            )
        # Predict branch deltas. Static state is copied exactly once.
        output = departure + value
        parts = split_branches(output, self.cfg.branch_sizes)
        parts["static"] = split_branches(departure, self.cfg.branch_sizes)["static"]
        return torch.cat(tuple(parts[name] for name in BRANCHES), dim=-2)

    @torch.no_grad()
    def sample(self, departure, ego_action, opponent_plan, steps=None, noise=None):
        noise = noise.clone() if noise is not None else None
        return self.transition(departure, ego_action, opponent_plan, steps, noise)


@register("model", "causal_world_action_dynamics")
def build_causal_world_action_dynamics(
    action_dim=320, opponent_plan_dim=320, device="cpu", **kwargs
):
    """Factory: ``model.type: causal_world_action_dynamics``.

    Builds only the trainable factorized flow.  The frozen belief encoder and
    tokenizers are supplied separately by the trainer (they come from the
    stage-1 ``belief_encoder_ckpt``, not from ``cfg.model``).
    """
    cfg = FactorizedDynamicsConfig.from_dict(kwargs.get("factorized_dynamics", kwargs))
    model = FactorizedWorldActionDynamics(int(action_dim), int(opponent_plan_dim), cfg)
    return model.to(device)
