"""Trainable world-action dynamics wrapped around frozen belief/tokenizer front ends.

``WorldActionDynamicsModule`` is the module the ``causal_world_action_dynamics``
stage trains.  It owns:

- the frozen ego / self-action / opponent tokenizers and the frozen
  ``WorldActionBeliefEncoder`` promoted from stage 1 (used only to *produce* the
  belief targets the flow transports between), and
- the trainable ``FactorizedWorldActionDynamics`` flow plus fresh readout heads
  (event / scalar / self-inverse / opponent-inverse) that decode the *predicted*
  next belief.

The forward pass tokenizes a batch (all under ``no_grad``) into the per-step
belief tokens, ego-action tokens, and opponent-plan tokens the dynamics loss
consumes.  Static branch immutability and the intrinsic/extrinsic/interaction
factorization live in ``FactorizedWorldActionDynamics``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import FactorizedDynamicsConfig, PredictiveBeliefConfig
from .dynamics import FactorizedWorldActionDynamics
from .encoder import TokenQueryHead, WorldActionBeliefEncoder


class WorldActionDynamicsModule(nn.Module):
    def __init__(
        self,
        ego_tokenizer,
        self_action_tokenizer,
        opponent_tokenizer,
        encoder,
        belief_cfg=None,
        dynamics_cfg=None,
    ):
        super().__init__()
        self.belief_cfg = belief_cfg or PredictiveBeliefConfig()
        self.dyn_cfg = dynamics_cfg or FactorizedDynamicsConfig()
        self.ego_tokenizer = ego_tokenizer.requires_grad_(False).eval()
        self.self_action_tokenizer = self_action_tokenizer.requires_grad_(False).eval()
        self.opponent_tokenizer = opponent_tokenizer.requires_grad_(False).eval()
        self.encoder = encoder.requires_grad_(False).eval()

        dlat = self.dyn_cfg.d_latent
        self.dynamics = FactorizedWorldActionDynamics(
            self_action_tokenizer.cfg.d_latent,
            opponent_tokenizer.cfg.d_latent,
            self.dyn_cfg,
        )
        # Fresh trainable readouts off the *predicted* next belief.
        self.event_head = TokenQueryHead(
            ego_tokenizer.n_spatial,
            dlat,
            self.belief_cfg.event_dim,
            self.belief_cfg.d_model,
            self.belief_cfg.n_heads,
        )
        self.scalar_head = nn.Sequential(
            nn.LayerNorm(dlat),
            nn.Linear(dlat, self.belief_cfg.d_model),
            nn.SiLU(),
            nn.Linear(self.belief_cfg.d_model, 3),
        )
        self.self_inverse = nn.Sequential(
            nn.LayerNorm(2 * dlat),
            nn.Linear(2 * dlat, self.belief_cfg.d_model),
            nn.SiLU(),
            nn.Linear(self.belief_cfg.d_model, self_action_tokenizer.cfg.d_latent),
        )
        self.opponent_inverse = TokenQueryHead(
            opponent_tokenizer.cfg.n_plan_tokens,
            2 * dlat,
            opponent_tokenizer.cfg.d_latent,
            self.belief_cfg.d_model,
            self.belief_cfg.n_heads,
        )

    def train(self, mode=True):
        super().train(mode)
        self.ego_tokenizer.eval()
        self.self_action_tokenizer.eval()
        self.opponent_tokenizer.eval()
        self.encoder.eval()
        return self

    @torch.no_grad()
    def encode_actions(self, batch, action_key):
        action, _, valid, _ = self.self_action_tokenizer(
            batch["local_obs"], batch[action_key]
        )
        return action, valid

    @torch.no_grad()
    def forward(self, batch):
        """Tokenize a batch into frozen belief / action / plan tensors."""
        spatial, _, visibility = self.ego_tokenizer.encode(
            batch["local_obs"], batch["local_visibility"]
        )
        action, valid = self.encode_actions(batch, "action")
        # The belief at t may consume actions only through t-1 (matches the
        # stage-1 encoder's causal history convention).
        history_action = torch.zeros_like(action)
        history_valid = torch.zeros_like(valid)
        history_action[:, 1:] = action[:, :-1]
        history_valid[:, 1:] = valid[:, :-1]
        is_first = batch.get("is_first")
        if is_first is not None:
            history_valid = history_valid & ~is_first[..., None].bool()
        belief = self.encoder(spatial, visibility, history_action, history_valid)["tokens"]
        plan = self.opponent_tokenizer.encode(batch["state"], batch["opponent_action"])[0]
        return {
            "belief": belief,
            "action": action,
            "valid": valid,
            "plan": plan,
        }

    # Readouts off a predicted next-belief token sequence -------------------
    def scalar_predictions(self, tokens):
        return self.scalar_head(tokens.mean(-2))
