from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossBlock
from .config import DirectDynamicsConfig
from .encoder import BRANCHES, split_branches


class DirectWorldActionDynamics(nn.Module):
    """One-pass, identity-residual transition with explicit action/intent memory."""

    def __init__(self, action_dim=320, plan_dim=320, cfg=None):
        super().__init__()
        self.cfg = cfg or DirectDynamicsConfig()
        d = self.cfg.d_model
        self.state_in = nn.Linear(self.cfg.d_latent, d)
        self.action_in = nn.Linear(action_dim, d)
        self.plan_in = nn.Linear(plan_dim, d)
        self.state_position = nn.Parameter(torch.zeros(self.cfg.n_tokens, d))
        self.action_type = nn.Parameter(torch.zeros(d))
        self.plan_type = nn.Parameter(torch.zeros(d))
        self.null_action = nn.Parameter(torch.zeros(d))
        self.null_plan = nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.state_position, std=0.02)
        nn.init.normal_(self.action_type, std=0.02)
        nn.init.normal_(self.plan_type, std=0.02)
        nn.init.normal_(self.null_action, std=0.02)
        nn.init.normal_(self.null_plan, std=0.02)
        self.blocks = nn.ModuleList(
            CrossBlock(
                d,
                self.cfg.n_heads,
                self.cfg.mlp_ratio,
                self.cfg.dropout,
            )
            for _ in range(self.cfg.depth)
        )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, self.cfg.d_latent)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, departure, action, action_valid, opponent_plan):
        lead = departure.shape[:-2]
        query = self.state_in(departure) + self.state_position
        action_memory = self.action_in(action)
        action_memory = torch.where(
            action_valid[..., None],
            action_memory + self.action_type,
            self.null_action,
        )
        plan_memory = self.plan_in(opponent_plan)
        if self.cfg.condition_on_opponent_intent:
            plan_memory = plan_memory + self.plan_type
        else:
            plan_memory = self.null_plan.expand_as(plan_memory)
        memory = torch.cat((query, action_memory, plan_memory), dim=-2)
        query = query.reshape(-1, self.cfg.n_tokens, self.cfg.d_model)
        memory = memory.reshape(-1, memory.shape[-2], self.cfg.d_model)
        for block in self.blocks:
            query = block(query, memory)
        correction = self.out(self.norm(query)).reshape(
            *lead, self.cfg.n_tokens, self.cfg.d_latent
        )
        predicted = departure + correction
        parts = split_branches(predicted, self.cfg.branch_sizes)
        parts["static"] = split_branches(departure, self.cfg.branch_sizes)["static"]
        return torch.cat(tuple(parts[name] for name in BRANCHES), dim=-2)
