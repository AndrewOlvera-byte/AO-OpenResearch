from __future__ import annotations

import torch
import torch.nn as nn

from .action_tokenizer import SelfActionTokenizer
from .config import (
    BeliefDynamicsConfig,
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    JointFlowConfig,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
)
from .ego_tokenizer import EgoObservationTokenizer
from .history_flow import (
    CausalHistoryTransformer,
    ConditionalBeliefDynamicsFlow,
    FutureStatePredictor,
    JointBeliefIntentFlow,
)
from .intent_prior import MultimodalOpponentIntentPrior
from .opponent_tokenizer import OpponentPlanTokenizer


class IncompleteInformationWorldModel(nn.Module):
    """Belief/intent inference in front of a separately trained mechanics model."""

    def __init__(
        self,
        full_state_tokenizer,
        grid_hw=(16, 16),
        *,
        ego_cfg=None,
        self_action_cfg=None,
        opponent_cfg=None,
        history_cfg=None,
        flow_cfg=None,
        mechanics=None,
    ):
        super().__init__()
        ego_cfg = ego_cfg or EgoTokenizerConfig()
        self_action_cfg = self_action_cfg or SelfActionTokenizerConfig()
        opponent_cfg = opponent_cfg or OpponentPlanTokenizerConfig()
        history_cfg = history_cfg or HistoryConfig()
        flow_cfg = flow_cfg or JointFlowConfig()
        self.full_state_tokenizer = full_state_tokenizer
        self.ego_tokenizer = EgoObservationTokenizer(grid_hw, ego_cfg)
        self.self_action_tokenizer = SelfActionTokenizer(
            full_state_tokenizer.n_tokens, grid_hw, self_action_cfg
        )
        self.opponent_tokenizer = OpponentPlanTokenizer(
            full_state_tokenizer.n_tokens, grid_hw, opponent_cfg
        )
        self.history = CausalHistoryTransformer(
            self.ego_tokenizer.n_spatial,
            self_action_cfg.max_action_events,
            ego_cfg.d_latent,
            history_cfg,
        )
        self.flow = JointBeliefIntentFlow(
            full_state_tokenizer.n_tokens,
            opponent_cfg.n_plan_tokens,
            full_state_tokenizer.d_latent,
            history_cfg.d_model,
            flow_cfg,
        )
        self.future_predictor = FutureStatePredictor(
            history_cfg.d_model,
            full_state_tokenizer.d_latent,
            full_state_tokenizer.n_tokens,
            history_cfg.n_heads,
        )
        self.mechanics = mechanics
        latent_dim = full_state_tokenizer.d_latent
        plan_dim = opponent_cfg.d_latent
        self.register_buffer("state_mean", torch.zeros(latent_dim))
        self.register_buffer("state_std", torch.ones(latent_dim))
        self.register_buffer("plan_mean", torch.zeros(plan_dim))
        self.register_buffer("plan_std", torch.ones(plan_dim))

    def freeze_teachers(self):
        for module in (
            self.full_state_tokenizer,
            self.ego_tokenizer,
            self.self_action_tokenizer,
            self.opponent_tokenizer,
            self.mechanics,
        ):
            if module is not None:
                module.requires_grad_(False)
                module.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.full_state_tokenizer.eval()
        self.ego_tokenizer.eval()
        self.self_action_tokenizer.eval()
        self.opponent_tokenizer.eval()
        if self.mechanics is not None:
            self.mechanics.eval()
        return self

    @torch.no_grad()
    def set_latent_stats(self, state_mean, state_std, plan_mean=None, plan_std=None):
        self.state_mean.copy_(state_mean.reshape_as(self.state_mean))
        self.state_std.copy_(state_std.reshape_as(self.state_std).clamp_min(1e-5))
        if plan_mean is not None:
            self.plan_mean.copy_(plan_mean.reshape_as(self.plan_mean))
        if plan_std is not None:
            self.plan_std.copy_(plan_std.reshape_as(self.plan_std).clamp_min(1e-5))

    def normalize_state(self, value):
        return (value - self.state_mean) / self.state_std

    def denormalize_state(self, value):
        return value * self.state_std + self.state_mean

    def normalize_plan(self, value):
        return (value - self.plan_mean) / self.plan_std

    def denormalize_plan(self, value):
        return value * self.plan_std + self.plan_mean

    def encode_history(self, batch):
        spatial, _, _ = self.ego_tokenizer.encode(
            batch["local_obs"], batch["local_visibility"]
        )
        current_action, _, current_valid, _ = self.self_action_tokenizer(
            batch["local_obs"], batch["action"]
        )
        action_tokens = torch.zeros_like(current_action)
        action_valid = torch.zeros_like(current_valid)
        action_tokens[:, 1:] = current_action[:, :-1]
        action_valid[:, 1:] = current_valid[:, :-1]
        is_first = batch.get("is_first")
        if is_first is not None:
            action_valid = action_valid & ~is_first[..., None].bool()
        return self.history(spatial, action_tokens, action_valid, is_first)

    def split_joint(self, value):
        return value.split((self.flow.n_state_tokens, self.flow.n_plan_tokens), dim=-2)

    @torch.no_grad()
    def sample_belief_intent(self, batch, steps=None, noise=None):
        registers = self.encode_history(batch)["registers"]
        joint = self.flow.sample(registers, steps=steps, noise=noise)
        state, plan = self.split_joint(joint)
        return {
            "state_tokens": self.denormalize_state(state),
            "plan_tokens": self.denormalize_plan(plan),
            "history_registers": registers,
        }


class OpponentIntentPriorModel(nn.Module):
    """Deployable history encoder with a frozen privileged plan teacher."""

    def __init__(
        self,
        full_state_tokenizer,
        grid_hw=(16, 16),
        *,
        ego_cfg=None,
        self_action_cfg=None,
        opponent_cfg=None,
        history_cfg=None,
        intent_cfg=None,
    ):
        super().__init__()
        ego_cfg = ego_cfg or EgoTokenizerConfig()
        self_action_cfg = self_action_cfg or SelfActionTokenizerConfig()
        opponent_cfg = opponent_cfg or OpponentPlanTokenizerConfig()
        history_cfg = history_cfg or HistoryConfig()
        intent_cfg = intent_cfg or IntentPriorConfig()
        self.full_state_tokenizer = full_state_tokenizer
        self.ego_tokenizer = EgoObservationTokenizer(grid_hw, ego_cfg)
        self.self_action_tokenizer = SelfActionTokenizer(
            full_state_tokenizer.n_tokens, grid_hw, self_action_cfg
        )
        self.opponent_tokenizer = OpponentPlanTokenizer(
            full_state_tokenizer.n_tokens, grid_hw, opponent_cfg
        )
        self.history = CausalHistoryTransformer(
            self.ego_tokenizer.n_spatial,
            self_action_cfg.max_action_events,
            ego_cfg.d_latent,
            history_cfg,
        )
        self.intent_prior = MultimodalOpponentIntentPrior(
            history_cfg.d_model,
            opponent_cfg.n_plan_tokens,
            opponent_cfg.d_latent,
            intent_cfg,
        )
        self.register_buffer("plan_mean", torch.zeros(opponent_cfg.d_latent))
        self.register_buffer("plan_std", torch.ones(opponent_cfg.d_latent))

    def freeze_teachers(self):
        for module in (
            self.full_state_tokenizer,
            self.ego_tokenizer,
            self.self_action_tokenizer,
            self.opponent_tokenizer,
        ):
            module.requires_grad_(False).eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.full_state_tokenizer.eval()
        self.ego_tokenizer.eval()
        self.self_action_tokenizer.eval()
        self.opponent_tokenizer.eval()
        return self

    @torch.no_grad()
    def set_plan_stats(self, mean, std):
        self.plan_mean.copy_(mean.reshape_as(self.plan_mean))
        self.plan_std.copy_(std.reshape_as(self.plan_std).clamp_min(1e-5))

    def normalize_plan(self, value):
        return (value - self.plan_mean) / self.plan_std

    def denormalize_plan(self, value):
        return value * self.plan_std + self.plan_mean

    def encode_history(self, batch):
        spatial, _, _ = self.ego_tokenizer.encode(
            batch["local_obs"], batch["local_visibility"]
        )
        current_action, _, current_valid, _ = self.self_action_tokenizer(
            batch["local_obs"], batch["action"]
        )
        action_tokens = torch.zeros_like(current_action)
        action_valid = torch.zeros_like(current_valid)
        action_tokens[:, 1:] = current_action[:, :-1]
        action_valid[:, 1:] = current_valid[:, :-1]
        is_first = batch.get("is_first")
        if is_first is not None:
            action_valid = action_valid & ~is_first[..., None].bool()
        return self.history(spatial, action_tokens, action_valid, is_first)

    def forward(self, batch):
        horizon = self.opponent_tokenizer.max_horizon
        anchors = batch["state"].shape[1] - horizon
        if anchors <= 0:
            raise ValueError("intent-prior sequence must exceed opponent horizon")
        registers = self.encode_history(batch)["registers"][:, :anchors]
        return (*self.intent_prior(registers), registers)

    @torch.no_grad()
    def sample_intent(self, batch, *, sample=True):
        plans, logits, _, registers = self(batch)
        selected, mode = self.intent_prior.select(plans, logits, sample=sample)
        return {
            "plan_tokens": self.denormalize_plan(selected),
            "mode": mode,
            "mode_probabilities": logits.softmax(-1),
            "all_plan_tokens": self.denormalize_plan(plans),
            "history_registers": registers,
        }


class BeliefDynamicsModel(nn.Module):
    """Next-world flow behind a frozen, deployable opponent-intent prior."""

    def __init__(self, intent_model, flow_cfg=None):
        super().__init__()
        self.intent_model = intent_model
        tokenizer = intent_model.full_state_tokenizer
        self.flow = ConditionalBeliefDynamicsFlow(
            tokenizer.n_tokens,
            tokenizer.d_latent,
            intent_model.history.cfg.d_model,
            intent_model.opponent_tokenizer.cfg.d_latent,
            intent_model.self_action_tokenizer.cfg.d_latent,
            flow_cfg or BeliefDynamicsConfig(),
            n_spatial_tokens=tokenizer.n_spatial,
            spatial_hw=(
                tokenizer.h // tokenizer.cfg.downsample,
                tokenizer.w // tokenizer.cfg.downsample,
            ),
        )
        self.register_buffer("state_mean", torch.zeros(tokenizer.d_latent))
        self.register_buffer("state_std", torch.ones(tokenizer.d_latent))

    @property
    def full_state_tokenizer(self):
        return self.intent_model.full_state_tokenizer

    def freeze_conditioner(self):
        self.intent_model.requires_grad_(False).eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        self.intent_model.eval()
        return self

    @torch.no_grad()
    def set_state_stats(self, mean, std):
        self.state_mean.copy_(mean.reshape_as(self.state_mean))
        self.state_std.copy_(std.reshape_as(self.state_std).clamp_min(1e-5))

    def normalize_state(self, value):
        return (value - self.state_mean) / self.state_std

    def denormalize_state(self, value):
        return value * self.state_std + self.state_mean

    @torch.no_grad()
    def current_belief(self, batch):
        """Return the deployable current-belief anchor when configured."""
        condition = self.encode_deploy_condition(batch, sample_intent=False)
        anchor = self.flow.current_anchor(condition["history_registers"])
        return self.denormalize_state(anchor)

    @torch.no_grad()
    def encode_condition(self, batch, *, sample_intent=False):
        plans, logits, _, registers = self.intent_model(batch)
        plan, mode = self.intent_model.intent_prior.select(
            plans, logits, sample=sample_intent
        )
        action, _, valid, _ = self.intent_model.self_action_tokenizer(
            batch["local_obs"], batch["action"]
        )
        anchors = registers.shape[1]
        return {
            "history_registers": registers,
            "plan_tokens": plan,
            "action_tokens": action[:, :anchors],
            "action_valid": valid[:, :anchors],
            "mode": mode,
            "mode_probabilities": logits.softmax(-1),
        }

    @torch.no_grad()
    def encode_deploy_condition(self, batch, *, sample_intent=False):
        """Condition every observed history row without the training-only trim.

        ``OpponentIntentPriorModel.forward`` drops the last ``max_horizon`` rows
        because its privileged plan targets need future opponent actions.  At
        deployment those targets do not exist and the most recent causal history
        row is precisely the one the actor needs.  This path uses the same frozen
        history and intent modules but keeps all causal rows.
        """
        registers = self.intent_model.encode_history(batch)["registers"]
        plans, logits, _ = self.intent_model.intent_prior(registers)
        plan, mode = self.intent_model.intent_prior.select(
            plans, logits, sample=sample_intent
        )
        action, _, valid, _ = self.intent_model.self_action_tokenizer(
            batch["local_obs"], batch["action"]
        )
        return {
            "history_registers": registers,
            "plan_tokens": plan,
            "action_tokens": action,
            "action_valid": valid,
            "mode": mode,
            "mode_probabilities": logits.softmax(-1),
        }

    @torch.no_grad()
    def sample_deploy_next(
        self, batch, *, sample_intent=True, steps=None, noise=None
    ):
        """Predict all deployment rows, including the latest observation row."""
        condition = self.encode_deploy_condition(
            batch, sample_intent=sample_intent
        )
        state = self.flow.sample(
            condition["history_registers"],
            condition["plan_tokens"],
            condition["action_tokens"],
            condition["action_valid"],
            steps=steps,
            noise=noise,
        )
        return {"state_tokens": self.denormalize_state(state), **condition}

    @torch.no_grad()
    def sample_next(self, batch, *, sample_intent=True, steps=None, noise=None):
        condition = self.encode_condition(batch, sample_intent=sample_intent)
        state = self.flow.sample(
            condition["history_registers"],
            condition["plan_tokens"],
            condition["action_tokens"],
            condition["action_valid"],
            steps=steps,
            noise=noise,
        )
        return {
            "state_tokens": self.denormalize_state(state),
            **condition,
        }
