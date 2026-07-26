"""Direct causal-paired dynamics behind frozen belief and opponent-intent encoders."""

from __future__ import annotations

import torch

from core.registry import register
from entrypoints.incomplete_info_common import (
    load_full_state_tokenizer,
    load_stage_weights,
    make_loaders,
    resolve_path,
)
from models.incomplete_info import (
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    OpponentIntentPriorModel,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
)
from models.incomplete_info.world_action import (
    DirectDynamicsConfig,
    DirectWorldActionDynamicsModule,
    PredictiveBeliefConfig,
    WorldActionBeliefEncoder,
)

from .PretrainTrainer import PretrainTrainer


@register("trainer", "direct_causal_world_action_dynamics")
class DirectWorldActionDynamicsTrainer(PretrainTrainer):
    phase = "direct_world_action"
    task = "incomplete_dynamics_paired"
    loss_type = "direct_causal_world_action_dynamics"

    def build_loaders(self):
        training = self.cfg.training or {}
        self.belief_path = resolve_path(training["belief_encoder_ckpt"])
        self.belief_checkpoint = torch.load(
            self.belief_path, map_location="cpu", weights_only=False
        )
        self.intent_path = resolve_path(training["intent_prior_ckpt"])
        self.intent_checkpoint = torch.load(
            self.intent_path, map_location="cpu", weights_only=False
        )
        self.belief_cfg = PredictiveBeliefConfig.from_dict(
            self.belief_checkpoint["predictive_belief_cfg"]
        )
        self.dyn_cfg = DirectDynamicsConfig.from_dict(
            (self.cfg.model or {}).get("direct_dynamics")
        )
        seq_len = int(training.get("seq_len", self.belief_cfg.context_length))
        task = (
            "incomplete_dynamics_paired_cf_belief"
            if bool(training.get("exact_counterfactual_belief", False))
            else self.task
        )
        return make_loaders(self.cfg, self.args, task=task, seq_len=seq_len)

    def load_frozen_teachers(self):
        ck = self.intent_checkpoint
        teacher, _, _ = load_full_state_tokenizer(
            ck["full_state_tokenizer_ckpt"], self.dataset, self.device
        )
        self.ego_cfg = EgoTokenizerConfig.from_dict(ck["ego_tokenizer_cfg"])
        self.action_cfg = SelfActionTokenizerConfig.from_dict(
            ck["self_action_tokenizer_cfg"]
        )
        self.opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
            ck["opponent_tokenizer_cfg"]
        )
        self.history_cfg = HistoryConfig.from_dict(ck["history_cfg"])
        self.intent_cfg = IntentPriorConfig.from_dict(ck["intent_prior_cfg"])
        return {"teacher": teacher}

    def build_model(self):
        ck = self.intent_checkpoint
        intent = OpponentIntentPriorModel(
            self.frozen["teacher"],
            self.grid_hw,
            ego_cfg=self.ego_cfg,
            self_action_cfg=self.action_cfg,
            opponent_cfg=self.opponent_cfg,
            history_cfg=self.history_cfg,
            intent_cfg=self.intent_cfg,
        ).to(self.device)
        load_stage_weights(
            intent.ego_tokenizer, ck["ego_tokenizer_ckpt"], ("tokenizer.", "")
        )
        load_stage_weights(
            intent.self_action_tokenizer, ck["self_action_tokenizer_ckpt"]
        )
        load_stage_weights(intent.opponent_tokenizer, ck["opponent_tokenizer_ckpt"])
        load_stage_weights(intent, self.intent_path)
        intent.freeze_teachers().requires_grad_(False).eval()

        encoder = WorldActionBeliefEncoder(
            intent.ego_tokenizer.cfg.d_latent,
            intent.self_action_tokenizer.cfg.d_latent,
            self.belief_cfg,
        ).to(self.device)
        load_stage_weights(encoder, self.belief_path, ("encoder.",))
        encoder.requires_grad_(False).eval()
        return DirectWorldActionDynamicsModule(
            intent.ego_tokenizer,
            intent.self_action_tokenizer,
            encoder,
            intent,
            self.belief_cfg,
            self.dyn_cfg,
        ).to(self.device)

    def build_metadata(self):
        return {
            "direct_dynamics_cfg": self.dyn_cfg.__dict__,
            "predictive_belief_cfg": self.belief_cfg.__dict__,
            "belief_encoder_ckpt": str(self.belief_path),
            "intent_prior_ckpt": str(self.intent_path),
            "condition_on_opponent_intent": (
                self.dyn_cfg.condition_on_opponent_intent
            ),
            "grid_hw": self.grid_hw,
            "data": str(self.data_path),
            "architecture": "CausalWorldAction-v1/direct-residual",
        }

    def checkpoint_policy(self):
        return (True, ())
