"""Registered trainer for the ``causal_world_action_dynamics`` stage.

Loads the frozen belief encoder + tokenizers promoted by stage 1
(``pretrain_world_action_encoder``) from ``training.belief_encoder_ckpt`` — whose
saved metadata carries every sub-config and upstream tokenizer path — then trains
the factorized flow dynamics on top.  The training loop, loss composition,
checkpointing, and eval all come from :class:`PretrainTrainer`.
"""

from __future__ import annotations

import torch

from core.registry import register
from entrypoints.incomplete_info_common import (
    load_full_state_tokenizer,
    load_stage_weights,
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
    FactorizedDynamicsConfig,
    PredictiveBeliefConfig,
    WorldActionBeliefEncoder,
    WorldActionDynamicsModule,
)

from .PretrainTrainer import PretrainTrainer


@register("trainer", "causal_world_action_dynamics")
class WorldActionDynamicsTrainer(PretrainTrainer):
    phase = "world_action_dynamics"
    task = "incomplete_dynamics_paired"
    default_seq_len = 64

    def load_frozen_teachers(self):
        training = self.cfg.training or {}
        belief_path = resolve_path(training["belief_encoder_ckpt"])
        ckpt = torch.load(belief_path, map_location="cpu", weights_only=False)
        self.belief_cfg = PredictiveBeliefConfig.from_dict(ckpt["predictive_belief_cfg"])
        ego_cfg = EgoTokenizerConfig.from_dict(ckpt["ego_tokenizer_cfg"])
        action_cfg = SelfActionTokenizerConfig.from_dict(ckpt["self_action_tokenizer_cfg"])
        opponent_cfg = OpponentPlanTokenizerConfig.from_dict(ckpt["opponent_tokenizer_cfg"])
        device, dataset = self.device, self.dataset

        full_tokenizer, _, _ = load_full_state_tokenizer(
            ckpt["full_state_tokenizer_ckpt"], dataset, device
        )
        ego = EgoObservationTokenizer(dataset.grid_hw, ego_cfg).to(device)
        action = SelfActionTokenizer(
            full_tokenizer.n_tokens, dataset.grid_hw, action_cfg
        ).to(device)
        opponent = OpponentPlanTokenizer(
            full_tokenizer.n_tokens, dataset.grid_hw, opponent_cfg
        ).to(device)
        load_stage_weights(ego, ckpt["ego_tokenizer_ckpt"], ("tokenizer.", ""))
        load_stage_weights(action, ckpt["self_action_tokenizer_ckpt"])
        load_stage_weights(opponent, ckpt["opponent_tokenizer_ckpt"])

        encoder = WorldActionBeliefEncoder(
            ego.cfg.d_latent, action.cfg.d_latent, self.belief_cfg
        ).to(device)
        load_stage_weights(encoder, belief_path, ("encoder.",))

        self._belief_ckpt_path = str(belief_path)
        return {
            "ego": ego,
            "self_action": action,
            "opponent": opponent,
            "encoder": encoder,
        }

    def build_model(self):
        dyn_cfg = FactorizedDynamicsConfig.from_dict(
            (self.cfg.model or {}).get("factorized_dynamics")
        )
        self.dyn_cfg = dyn_cfg
        return WorldActionDynamicsModule(
            self.frozen["ego"],
            self.frozen["self_action"],
            self.frozen["opponent"],
            self.frozen["encoder"],
            self.belief_cfg,
            dyn_cfg,
        ).to(self.device)

    def build_metadata(self):
        return {
            "factorized_dynamics_cfg": self.dyn_cfg.__dict__,
            "predictive_belief_cfg": self.belief_cfg.__dict__,
            "belief_encoder_ckpt": self._belief_ckpt_path,
            "grid_hw": self.grid_hw,
            "data": str(self.data_path),
            "architecture": "CausalWorldAction-v1/factorized-dynamics",
        }

    def checkpoint_policy(self):
        # Save only the trainable flow + readout heads; the frozen front end is
        # reconstructed from ``belief_encoder_ckpt`` at load time.
        return (True, ())
