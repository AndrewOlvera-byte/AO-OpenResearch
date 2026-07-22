"""Registered trainers for the incomplete-information belief / dynamics stages.

``OpponentIntentPriorTrainer`` -> ``BeliefDynamicsTrainer`` -> the joint-flow world
model.  Each subclass keeps the exact frozen-teacher wiring, latent-stat plumbing,
and loss-coefficient mapping of the original entrypoint; the training loop comes
from :class:`PretrainTrainer`.
"""

from __future__ import annotations

import torch

from core.registry import register
from entrypoints.incomplete_info_common import (
    load_frozen_mechanics,
    load_full_state_tokenizer,
    load_stage_weights,
    make_loaders,
    measure_plan_stats,
    resolve_path,
)
from models.incomplete_info import (
    BeliefDynamicsConfig,
    BeliefDynamicsModel,
    EgoTokenizerConfig,
    HistoryConfig,
    IncompleteInformationWorldModel,
    IntentPriorConfig,
    JointFlowConfig,
    OpponentIntentPriorModel,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
    belief_dynamics_loss,
    joint_flow_world_model_loss,
    opponent_intent_prior_loss,
)

from .PretrainTrainer import PretrainTrainer


@register("trainer", "opponent_intent_prior")
class OpponentIntentPriorTrainer(PretrainTrainer):
    phase = "intent"
    task = "incomplete_dynamics"

    def build_loaders(self):
        model_cfg = self.cfg.model or {}
        self.ego_cfg = EgoTokenizerConfig.from_dict(model_cfg.get("ego_tokenizer"))
        self.action_cfg = SelfActionTokenizerConfig.from_dict(
            model_cfg.get("self_action_tokenizer")
        )
        self.opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
            model_cfg.get("opponent_tokenizer")
        )
        self.history_cfg = HistoryConfig.from_dict(model_cfg.get("history"))
        self.intent_cfg = IntentPriorConfig.from_dict(model_cfg.get("intent_prior"))
        seq_len = int((self.cfg.training or {}).get("seq_len", self.history_cfg.context_length))
        return make_loaders(self.cfg, self.args, task=self.task, seq_len=seq_len)

    def load_frozen_teachers(self):
        teacher, teacher_cfg, _ = load_full_state_tokenizer(
            (self.cfg.training or {})["full_state_tokenizer_ckpt"], self.dataset, self.device
        )
        self.teacher_cfg = teacher_cfg
        return {"teacher": teacher}

    def build_model(self):
        training = self.cfg.training or {}
        model = OpponentIntentPriorModel(
            self.frozen["teacher"],
            self.grid_hw,
            ego_cfg=self.ego_cfg,
            self_action_cfg=self.action_cfg,
            opponent_cfg=self.opponent_cfg,
            history_cfg=self.history_cfg,
            intent_cfg=self.intent_cfg,
        ).to(self.device)
        load_stage_weights(model.ego_tokenizer, training["ego_tokenizer_ckpt"], ("tokenizer.", ""))
        load_stage_weights(model.self_action_tokenizer, training["self_action_tokenizer_ckpt"])
        load_stage_weights(model.opponent_tokenizer, training["opponent_tokenizer_ckpt"])
        plan_mean, plan_std = measure_plan_stats(
            model.opponent_tokenizer, self.train_loader, self.device,
            batches=training.get("stats_batches", 32),
        )
        model.set_plan_stats(plan_mean, plan_std)
        model.freeze_teachers()
        return model

    def build_loss(self):
        training = self.cfg.training or {}
        coef = training.get("loss", {})
        model = self.model

        def loss_fn(batch):
            return opponent_intent_prior_loss(
                model, batch,
                latent_coef=coef.get("latent", 1.0),
                event_coef=coef.get("event", 1.0),
                mode_coef=coef.get("mode", 0.25),
                balance_coef=coef.get("balance", 0.1),
                contrastive_coef=coef.get("contrastive", 0.5),
                shuffled_margin_coef=coef.get("shuffled_margin", 0.5),
                diversity_coef=coef.get("diversity", 0.05),
                shuffled_margin=training.get("shuffled_history_margin", 0.25),
                diversity_floor=training.get("diversity_floor", 0.75),
                contrastive_temperature=training.get("contrastive_temperature", 0.1),
            )

        return loss_fn

    def checkpoint_policy(self):
        return (True, ())

    def build_metadata(self):
        training = self.cfg.training or {}
        return {
            "ego_tokenizer_cfg": self.ego_cfg.__dict__,
            "self_action_tokenizer_cfg": self.action_cfg.__dict__,
            "opponent_tokenizer_cfg": self.opponent_cfg.__dict__,
            "history_cfg": self.history_cfg.__dict__,
            "intent_prior_cfg": self.intent_cfg.__dict__,
            "tokenizer_cfg": self.teacher_cfg.__dict__,
            "grid_hw": self.grid_hw,
            "full_state_tokenizer_ckpt": training["full_state_tokenizer_ckpt"],
            "ego_tokenizer_ckpt": training["ego_tokenizer_ckpt"],
            "self_action_tokenizer_ckpt": training["self_action_tokenizer_ckpt"],
            "opponent_tokenizer_ckpt": training["opponent_tokenizer_ckpt"],
            "data": str(self.data_path),
        }


@register("trainer", "belief_dynamics")
class BeliefDynamicsTrainer(PretrainTrainer):
    phase = "belief_dynamics"

    def build_loaders(self):
        training = self.cfg.training or {}
        self.flow_cfg = BeliefDynamicsConfig.from_dict((self.cfg.model or {}).get("belief_dynamics"))
        self.intent_path = resolve_path(training["intent_prior_ckpt"])
        self.intent_checkpoint = torch.load(
            self.intent_path, map_location="cpu", weights_only=False
        )
        self.history_cfg = HistoryConfig.from_dict(self.intent_checkpoint["history_cfg"])
        self.opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
            self.intent_checkpoint["opponent_tokenizer_cfg"]
        )
        seq_len = int(training.get("seq_len", self.history_cfg.context_length))
        paired = float(training.get("loss", {}).get("counterfactual", 0.0)) > 0.0
        task = "incomplete_dynamics_paired" if paired else "incomplete_dynamics"
        return make_loaders(self.cfg, self.args, task=task, seq_len=seq_len)

    def load_frozen_teachers(self):
        training = self.cfg.training or {}
        ck = self.intent_checkpoint
        full_path = training.get("full_state_tokenizer_ckpt", ck["full_state_tokenizer_ckpt"])
        teacher, teacher_cfg, _ = load_full_state_tokenizer(full_path, self.dataset, self.device)
        self.teacher_cfg = teacher_cfg
        self.full_tokenizer_path = full_path
        self.ego_cfg = EgoTokenizerConfig.from_dict(ck["ego_tokenizer_cfg"])
        self.action_cfg = SelfActionTokenizerConfig.from_dict(ck["self_action_tokenizer_cfg"])
        self.intent_cfg = IntentPriorConfig.from_dict(ck["intent_prior_cfg"])
        return {"teacher": teacher}

    def build_model(self):
        training = self.cfg.training or {}
        ck = self.intent_checkpoint
        intent_model = OpponentIntentPriorModel(
            self.frozen["teacher"], self.grid_hw,
            ego_cfg=self.ego_cfg, self_action_cfg=self.action_cfg,
            opponent_cfg=self.opponent_cfg, history_cfg=self.history_cfg,
            intent_cfg=self.intent_cfg,
        ).to(self.device)
        load_stage_weights(
            intent_model.ego_tokenizer,
            training.get("ego_tokenizer_ckpt", ck["ego_tokenizer_ckpt"]), ("tokenizer.", ""),
        )
        load_stage_weights(
            intent_model.self_action_tokenizer,
            training.get("self_action_tokenizer_ckpt", ck["self_action_tokenizer_ckpt"]),
        )
        load_stage_weights(
            intent_model.opponent_tokenizer,
            training.get("opponent_tokenizer_ckpt", ck["opponent_tokenizer_ckpt"]),
        )
        load_stage_weights(intent_model, self.intent_path)

        model = BeliefDynamicsModel(intent_model, self.flow_cfg).to(self.device)
        mechanics_path = resolve_path(training["mechanics_stats_ckpt"])
        mechanics = torch.load(mechanics_path, map_location="cpu", weights_only=False)
        state_mean = mechanics["model"].get("dynamics.latent_mean")
        state_std = mechanics["model"].get("dynamics.latent_std")
        if state_mean is None or state_std is None:
            raise ValueError("mechanics checkpoint lacks dynamics latent statistics")
        model.set_state_stats(state_mean.to(self.device), state_std.to(self.device))
        model.freeze_conditioner()
        self.mechanics_stats_path = mechanics_path

        init_from = training.get("init_from")
        if init_from:
            self._init_flow_from(model, resolve_path(init_from))
        self.trainable_scope = training.get("trainable_scope", "flow")
        self._apply_trainable_scope(model, self.trainable_scope)
        return model

    def _init_flow_from(self, model, init_path):
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        allowed_new = (
            "flow.action_residual_" if model.flow.explicit_action_residual
            else "__no_new_flow_parameters__"
        )
        bad_missing = [
            n for n in missing
            if not n.startswith("intent_model.") and not n.startswith(allowed_new)
        ]
        if bad_missing or unexpected:
            raise ValueError(
                f"{init_path}: incompatible initialization; "
                f"missing={bad_missing}, unexpected={unexpected}"
            )
        print(
            f"[belief-dynamics] initialized flow weights from {init_path} "
            f"at source step {initial.get('step')}",
            flush=True,
        )

    @staticmethod
    def _apply_trainable_scope(model, scope):
        if scope == "action_residual":
            if not model.flow.explicit_action_residual:
                raise ValueError(
                    "trainable_scope=action_residual requires "
                    "model.belief_dynamics.explicit_action_residual=true"
                )
            model.flow.requires_grad_(False)
            for parameter in model.flow.action_residual_parameters():
                parameter.requires_grad_(True)
        elif scope != "flow":
            raise ValueError(f"unknown belief dynamics trainable_scope {scope!r}")

    def build_loss(self):
        training = self.cfg.training or {}
        coef = training.get("loss", {})
        model = self.model

        def loss_fn(batch):
            return belief_dynamics_loss(
                model, batch,
                flow_coef=coef.get("flow", 1.0),
                prior_coef=coef.get("prior", 1.0),
                grounding_coef=coef.get("grounding", 0.25),
                history_rank_coef=coef.get("history_rank", 1.0),
                intent_rank_coef=coef.get("intent_rank", 1.0),
                action_rank_coef=coef.get("action_rank", 0.0),
                condition_margin=training.get("condition_margin", 0.05),
                visible_boost=training.get("visible_boost", 3.0),
                occupied_boost=training.get("occupied_boost", 0.0),
                hidden_occupied_boost=training.get("hidden_occupied_boost", 0.0),
                rank_anchors=training.get("rank_anchors", 1),
                action_residual_coef=coef.get("action_residual", 0.0),
                anchor_coef=coef.get("anchor", 0.0),
                anchor_grounding_coef=coef.get("anchor_grounding", 0.0),
                counterfactual_coef=coef.get("counterfactual", 0.0),
                counterfactual_effect_coef=coef.get("counterfactual_effect", 0.0),
            )

        return loss_fn

    def checkpoint_policy(self):
        prefixes = ("flow.",) if self.trainable_scope == "action_residual" else ()
        return (True, prefixes)

    def build_metadata(self):
        return {
            "belief_dynamics_cfg": self.flow_cfg.__dict__,
            "tokenizer_cfg": self.teacher_cfg.__dict__,
            "ego_tokenizer_cfg": self.ego_cfg.__dict__,
            "self_action_tokenizer_cfg": self.action_cfg.__dict__,
            "opponent_tokenizer_cfg": self.opponent_cfg.__dict__,
            "history_cfg": self.history_cfg.__dict__,
            "intent_prior_cfg": self.intent_cfg.__dict__,
            "grid_hw": self.grid_hw,
            "intent_prior_ckpt": str(self.intent_path),
            "mechanics_stats_ckpt": str(self.mechanics_stats_path),
            "full_state_tokenizer_ckpt": str(self.full_tokenizer_path),
            "data": str(self.data_path),
            "trainable_scope": self.trainable_scope,
        }


@register("trainer", "joint_flow_dynamics")
class JointFlowDynamicsTrainer(PretrainTrainer):
    phase = "joint"

    def build_loaders(self):
        model_cfg = self.cfg.model or {}
        self.ego_cfg = EgoTokenizerConfig.from_dict(model_cfg.get("ego_tokenizer"))
        self.action_cfg = SelfActionTokenizerConfig.from_dict(model_cfg.get("self_action_tokenizer"))
        self.opponent_cfg = OpponentPlanTokenizerConfig.from_dict(model_cfg.get("opponent_tokenizer"))
        self.history_cfg = HistoryConfig.from_dict(model_cfg.get("history"))
        self.flow_cfg = JointFlowConfig.from_dict(model_cfg.get("flow"))
        seq_len = int((self.cfg.training or {}).get(
            "seq_len",
            max(self.history_cfg.context_length, max(self.opponent_cfg.horizons) + 1),
        ))
        return make_loaders(self.cfg, self.args, task="incomplete_dynamics", seq_len=seq_len)

    def load_frozen_teachers(self):
        training = self.cfg.training or {}
        mechanics, mechanics_ckpt = load_frozen_mechanics(training["mechanics_ckpt"], self.device)
        self._mechanics_ckpt = mechanics_ckpt
        return {"mechanics": mechanics}

    def build_model(self):
        training = self.cfg.training or {}
        mechanics = self.frozen["mechanics"]
        world = IncompleteInformationWorldModel(
            mechanics.tokenizer, self.grid_hw,
            ego_cfg=self.ego_cfg, self_action_cfg=self.action_cfg,
            opponent_cfg=self.opponent_cfg, history_cfg=self.history_cfg,
            flow_cfg=self.flow_cfg, mechanics=mechanics,
        ).to(self.device)
        load_stage_weights(world.ego_tokenizer, training["ego_tokenizer_ckpt"], ("tokenizer.", ""))
        load_stage_weights(world.self_action_tokenizer, training["self_action_tokenizer_ckpt"])
        load_stage_weights(world.opponent_tokenizer, training["opponent_tokenizer_ckpt"])
        state_mean = self._mechanics_ckpt["model"].get("dynamics.latent_mean")
        state_std = self._mechanics_ckpt["model"].get("dynamics.latent_std")
        if state_mean is None or state_std is None:
            raise ValueError("mechanics checkpoint lacks dynamics latent statistics")
        plan_mean, plan_std = measure_plan_stats(
            world.opponent_tokenizer, self.train_loader, self.device,
            batches=training.get("stats_batches", 16),
        )
        world.set_latent_stats(state_mean.to(self.device), state_std.to(self.device), plan_mean, plan_std)
        world.freeze_teachers()
        return world

    def build_loss(self):
        coef = (self.cfg.training or {}).get("loss", {})
        world = self.model

        def loss_fn(batch):
            return joint_flow_world_model_loss(
                world, batch,
                flow_coef=coef.get("flow", 1.0),
                grounding_coef=coef.get("grounding", 0.25),
                opponent_event_coef=coef.get("opponent_event", 0.5),
                future_jepa_coef=coef.get("future_jepa", 0.25),
            )

        return loss_fn

    def checkpoint_policy(self):
        return (True, ())

    def build_metadata(self):
        training = self.cfg.training or {}
        return {
            "ego_tokenizer_cfg": self.ego_cfg.__dict__,
            "self_action_tokenizer_cfg": self.action_cfg.__dict__,
            "opponent_tokenizer_cfg": self.opponent_cfg.__dict__,
            "history_cfg": self.history_cfg.__dict__,
            "flow_cfg": self.flow_cfg.__dict__,
            "grid_hw": self.grid_hw,
            "mechanics_ckpt": training["mechanics_ckpt"],
            "ego_tokenizer_ckpt": training["ego_tokenizer_ckpt"],
            "self_action_tokenizer_ckpt": training["self_action_tokenizer_ckpt"],
            "opponent_tokenizer_ckpt": training["opponent_tokenizer_ckpt"],
            "data": str(self.data_path),
        }
