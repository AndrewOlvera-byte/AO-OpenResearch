"""Registered trainer for the ``causal_world_action_encoder`` (stage 1).

Builds the factorized predictive-belief encoder on top of the three frozen
tokenizers promoted by the opponent-intent stage (paths carried in
``source_intent_ckpt``'s metadata), trains it with the registry-composed
``predictive_belief`` loss, and maintains the EMA target encoder.
"""

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
    EgoObservationTokenizer,
    EgoTokenizerConfig,
    OpponentPlanTokenizer,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizer,
    SelfActionTokenizerConfig,
)
from models.incomplete_info.world_action import (
    PredictiveBeliefConfig,
    PredictiveBeliefPretrainer,
)

from .PretrainTrainer import PretrainTrainer


@register("trainer", "causal_world_action_encoder")
class WorldActionEncoderTrainer(PretrainTrainer):
    phase = "world_action_encoder"
    task = "incomplete_dynamics_paired"
    loss_type = "predictive_belief"

    def build_loaders(self):
        training = self.cfg.training or {}
        self.belief_cfg = PredictiveBeliefConfig.from_dict(
            (self.cfg.model or {}).get("predictive_belief")
        )
        self.source_path = resolve_path(training["source_intent_ckpt"])
        self.source = torch.load(self.source_path, map_location="cpu", weights_only=False)
        seq_len = int(training.get("seq_len", self.belief_cfg.context_length))
        return make_loaders(self.cfg, self.args, task=self.task, seq_len=seq_len)

    def load_frozen_teachers(self):
        training = self.cfg.training or {}
        source = self.source
        full_tokenizer, tokenizer_cfg, _ = load_full_state_tokenizer(
            training.get("full_state_tokenizer_ckpt", source["full_state_tokenizer_ckpt"]),
            self.dataset,
            self.device,
        )
        self.tokenizer_cfg = tokenizer_cfg
        self.ego_cfg = EgoTokenizerConfig.from_dict(source["ego_tokenizer_cfg"])
        self.action_cfg = SelfActionTokenizerConfig.from_dict(source["self_action_tokenizer_cfg"])
        self.opponent_cfg = OpponentPlanTokenizerConfig.from_dict(source["opponent_tokenizer_cfg"])
        ego = EgoObservationTokenizer(self.grid_hw, self.ego_cfg).to(self.device)
        action = SelfActionTokenizer(full_tokenizer.n_tokens, self.grid_hw, self.action_cfg).to(self.device)
        opponent = OpponentPlanTokenizer(full_tokenizer.n_tokens, self.grid_hw, self.opponent_cfg).to(self.device)
        load_stage_weights(
            ego, training.get("ego_tokenizer_ckpt", source["ego_tokenizer_ckpt"]), ("tokenizer.", "")
        )
        load_stage_weights(action, training.get("self_action_tokenizer_ckpt", source["self_action_tokenizer_ckpt"]))
        load_stage_weights(opponent, training.get("opponent_tokenizer_ckpt", source["opponent_tokenizer_ckpt"]))
        return {"ego": ego, "self_action": action, "opponent": opponent}

    def build_model(self):
        return PredictiveBeliefPretrainer(
            self.frozen["ego"], self.frozen["self_action"], self.frozen["opponent"], self.belief_cfg
        ).to(self.device)

    def after_step(self):
        self.model.update_target()

    def checkpoint_policy(self):
        return (True, ("target_encoder.",))

    def build_metadata(self):
        training = self.cfg.training or {}
        source = self.source
        return {
            "predictive_belief_cfg": self.belief_cfg.__dict__,
            "tokenizer_cfg": self.tokenizer_cfg.__dict__,
            "ego_tokenizer_cfg": self.ego_cfg.__dict__,
            "self_action_tokenizer_cfg": self.action_cfg.__dict__,
            "opponent_tokenizer_cfg": self.opponent_cfg.__dict__,
            "grid_hw": self.grid_hw,
            "source_intent_ckpt": str(self.source_path),
            "full_state_tokenizer_ckpt": training.get(
                "full_state_tokenizer_ckpt", source["full_state_tokenizer_ckpt"]
            ),
            "ego_tokenizer_ckpt": training.get("ego_tokenizer_ckpt", source["ego_tokenizer_ckpt"]),
            "self_action_tokenizer_ckpt": training.get(
                "self_action_tokenizer_ckpt", source["self_action_tokenizer_ckpt"]
            ),
            "opponent_tokenizer_ckpt": training.get(
                "opponent_tokenizer_ckpt", source["opponent_tokenizer_ckpt"]
            ),
            "data": str(self.data_path),
            "architecture": "CausalWorldAction-v1",
        }
