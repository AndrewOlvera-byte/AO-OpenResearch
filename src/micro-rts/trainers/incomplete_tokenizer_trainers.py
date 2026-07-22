"""Registered trainers for the incomplete-information tokenizer stages.

Each is a thin :class:`PretrainTrainer` subclass declaring how to build its
loaders / frozen full-state teacher / model / loss.  The loop, checkpointing,
and eval come from the base.  Resolved by config ``training.trainer`` (or
``model.type``) through the generic ``entrypoints/pretrain`` dispatcher.
"""

from __future__ import annotations

import torch

from core.registry import register
from entrypoints.incomplete_info_common import load_full_state_tokenizer, resolve_path
from models.incomplete_info import (
    EgoTokenizerConfig,
    EgoTokenizerPretrainer,
    OpponentPlanTokenizer,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizer,
    SelfActionTokenizerConfig,
    ego_tokenizer_loss,
    opponent_plan_tokenizer_loss,
    self_action_tokenizer_loss,
)

from .PretrainTrainer import PretrainTrainer


class _TeacherTokenizerTrainer(PretrainTrainer):
    """Shared: load the frozen full-state tokenizer used as the SSL teacher."""

    def load_frozen_teachers(self):
        training = self.cfg.training or {}
        teacher, teacher_cfg, _ = load_full_state_tokenizer(
            training["full_state_tokenizer_ckpt"],
            self.dataset,
            self.device,
            training.get("full_state_tokenizer_stats_ckpt"),
        )
        self.teacher_cfg = teacher_cfg
        return {"teacher": teacher}

    def build_metadata(self):
        return {
            "tokenizer_cfg": self.teacher_cfg.__dict__,
            "grid_hw": self.grid_hw,
            "data": str(self.data_path),
        }


@register("trainer", "incomplete_ego_tokenizer")
class EgoTokenizerTrainer(_TeacherTokenizerTrainer):
    phase = "ego"
    task = "incomplete_obs_tokenizer"
    default_seq_len = 1

    def build_model(self):
        cfg = EgoTokenizerConfig.from_dict((self.cfg.model or {}).get("ego_tokenizer"))
        if cfg.d_latent != self.frozen["teacher"].d_latent:
            raise ValueError("ego and full-state tokenizer latent dimensions must match")
        self.model_cfg = cfg
        return EgoTokenizerPretrainer(self.grid_hw, cfg).to(self.device)

    def build_loss(self):
        coef = (self.cfg.training or {}).get("loss", {})
        teacher, model = self.frozen["teacher"], self.model

        def loss_fn(batch):
            return ego_tokenizer_loss(
                model,
                batch,
                full_state_tokenizer=teacher,
                reconstruction_coef=coef.get("reconstruction", 1.0),
                visibility_coef=coef.get("visibility", 0.1),
                jepa_coef=coef.get("jepa", 0.25),
                teacher_coef=coef.get("full_teacher", 0.25),
            )

        return loss_fn

    def after_step(self):
        self.model.update_target()

    def build_metadata(self):
        return {"ego_tokenizer_cfg": self.model_cfg.__dict__, **super().build_metadata()}


@register("trainer", "incomplete_self_action_tokenizer")
class SelfActionTokenizerTrainer(_TeacherTokenizerTrainer):
    phase = "self_action"
    task = "incomplete_action_tokenizer"
    default_seq_len = 1

    def build_model(self):
        cfg = SelfActionTokenizerConfig.from_dict(
            (self.cfg.model or {}).get("self_action_tokenizer")
        )
        teacher = self.frozen["teacher"]
        if cfg.d_latent != teacher.d_latent:
            raise ValueError("action and state latent dimensions must match")
        self.model_cfg = cfg
        model = SelfActionTokenizer(teacher.n_tokens, self.grid_hw, cfg).to(self.device)
        init_ckpt = (self.cfg.training or {}).get("init_action_tokenizer_ckpt")
        if init_ckpt:
            self._init_event_encoder(model, init_ckpt)
        return model

    @staticmethod
    def _init_event_encoder(model, path):
        checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
        source = checkpoint.get("model", checkpoint)
        mapped = {
            key.removeprefix("action_encoder."): value
            for key, value in source.items()
            if key.startswith("action_encoder.")
        }
        if mapped:
            model.event_encoder.load_state_dict(mapped)

    def build_loss(self):
        coef = (self.cfg.training or {}).get("loss", {})
        teacher, model = self.frozen["teacher"], self.model

        def loss_fn(batch):
            return self_action_tokenizer_loss(
                model,
                teacher,
                batch,
                reconstruction_coef=coef.get("reconstruction", 1.0),
                forward_coef=coef.get("forward", 1.0),
                changed_token_boost=coef.get("changed_token_boost", 8.0),
            )

        return loss_fn

    def build_metadata(self):
        return {
            "self_action_tokenizer_cfg": self.model_cfg.__dict__,
            **super().build_metadata(),
        }


@register("trainer", "incomplete_opponent_plan_tokenizer")
class OpponentPlanTokenizerTrainer(_TeacherTokenizerTrainer):
    phase = "opponent_plan"
    task = "incomplete_opponent_tokenizer"

    def build_loaders(self):
        from entrypoints.incomplete_info_common import make_loaders

        cfg = OpponentPlanTokenizerConfig.from_dict(
            (self.cfg.model or {}).get("opponent_tokenizer")
        )
        self.model_cfg = cfg
        seq_len = int((self.cfg.training or {}).get("seq_len", max(cfg.horizons) + 1))
        return make_loaders(self.cfg, self.args, task=self.task, seq_len=seq_len)

    def build_model(self):
        teacher = self.frozen["teacher"]
        if self.model_cfg.d_latent != teacher.d_latent:
            raise ValueError("opponent plan and state latent dimensions must match")
        return OpponentPlanTokenizer(teacher.n_tokens, self.grid_hw, self.model_cfg).to(
            self.device
        )

    def build_loss(self):
        coef = (self.cfg.training or {}).get("loss", {})
        teacher, model = self.frozen["teacher"], self.model

        def loss_fn(batch):
            return opponent_plan_tokenizer_loss(
                model,
                teacher,
                batch,
                event_coef=coef.get("event", 1.0),
                future_state_coef=coef.get("future_state", 0.5),
            )

        return loss_fn

    def build_metadata(self):
        return {
            "opponent_tokenizer_cfg": self.model_cfg.__dict__,
            **super().build_metadata(),
        }
