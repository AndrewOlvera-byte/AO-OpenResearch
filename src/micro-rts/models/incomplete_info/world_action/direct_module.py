from __future__ import annotations

import torch
import torch.nn as nn

from .config import DirectDynamicsConfig, PredictiveBeliefConfig
from .direct_dynamics import DirectWorldActionDynamics
from .encoder import TokenQueryHead


class DirectWorldActionDynamicsModule(nn.Module):
    """Frozen incomplete-information representation plus direct residual mechanics."""

    def __init__(
        self,
        ego_tokenizer,
        self_action_tokenizer,
        belief_encoder,
        intent_model,
        belief_cfg=None,
        dynamics_cfg=None,
    ):
        super().__init__()
        self.belief_cfg = belief_cfg or PredictiveBeliefConfig()
        self.dyn_cfg = dynamics_cfg or DirectDynamicsConfig()
        self.ego_tokenizer = ego_tokenizer.requires_grad_(False).eval()
        self.self_action_tokenizer = self_action_tokenizer.requires_grad_(False).eval()
        self.belief_encoder = belief_encoder.requires_grad_(False).eval()
        self.intent_model = intent_model.requires_grad_(False).eval()
        self.dynamics = DirectWorldActionDynamics(
            self_action_tokenizer.cfg.d_latent,
            self.intent_model.opponent_tokenizer.cfg.d_latent,
            self.dyn_cfg,
        )
        dlat = self.dyn_cfg.d_latent
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

    def train(self, mode=True):
        super().train(mode)
        self.ego_tokenizer.eval()
        self.self_action_tokenizer.eval()
        self.belief_encoder.eval()
        self.intent_model.eval()
        return self

    @torch.no_grad()
    def encode_actions(self, batch, key="action"):
        action, _, valid, _ = self.self_action_tokenizer(
            batch["local_obs"], batch[key]
        )
        return action, valid

    @torch.no_grad()
    def forward(self, batch):
        spatial, _, visibility = self.ego_tokenizer.encode(
            batch["local_obs"], batch["local_visibility"]
        )
        action, valid = self.encode_actions(batch)
        history_action = torch.zeros_like(action)
        history_valid = torch.zeros_like(valid)
        history_action[:, 1:] = action[:, :-1]
        history_valid[:, 1:] = valid[:, :-1]
        if batch.get("is_first") is not None:
            history_valid &= ~batch["is_first"][..., None].bool()
        belief = self.belief_encoder(
            spatial, visibility, history_action, history_valid
        )["tokens"]
        history = self.intent_model.history(
            spatial,
            history_action,
            history_valid,
            batch.get("is_first"),
        )
        horizon = self.intent_model.opponent_tokenizer.max_horizon
        anchors = belief.shape[1] - horizon
        registers = history["registers"][:, :anchors]
        plans, logits, _ = self.intent_model.intent_prior(registers)
        plan, mode = self.intent_model.intent_prior.select(
            plans, logits, sample=False
        )
        length = min(belief.shape[1] - 1, plan.shape[1])
        return {
            "belief": belief,
            "spatial": spatial,
            "visibility": visibility,
            "history_action": history_action,
            "history_valid": history_valid,
            "action": action,
            "valid": valid,
            "plan": plan,
            "mode": mode,
            "mode_probabilities": logits.softmax(-1),
            "history_registers": registers,
            "length": length,
        }

    @torch.no_grad()
    def counterfactual_belief_target(self, batch, encoded, anchor):
        """Encode one exact cloned arrival without multiplying work by T.

        All batch rows use the same anchor, so this is one causal teacher pass.
        The prefix is factual and only the final observation/action pair is
        replaced, which represents the single intervention stored by the clone.
        """
        cf_spatial, _, cf_visibility = self.ego_tokenizer.encode(
            batch["counterfactual_local_obs"][:, anchor : anchor + 1],
            batch["counterfactual_local_visibility"][:, anchor : anchor + 1],
        )
        cf_action, cf_valid = self.encode_actions(batch, "counterfactual_action")
        spatial = torch.cat(
            (encoded["spatial"][:, : anchor + 1], cf_spatial), dim=1
        )
        visibility = torch.cat(
            (encoded["visibility"][:, : anchor + 1], cf_visibility), dim=1
        )
        history_action = torch.cat(
            (
                encoded["history_action"][:, : anchor + 1],
                cf_action[:, anchor : anchor + 1],
            ),
            dim=1,
        )
        history_valid = torch.cat(
            (
                encoded["history_valid"][:, : anchor + 1],
                cf_valid[:, anchor : anchor + 1],
            ),
            dim=1,
        )
        return self.belief_encoder(
            spatial, visibility, history_action, history_valid
        )["tokens"][:, -1]

    def scalar_predictions(self, tokens):
        return self.scalar_head(tokens.mean(-2))
