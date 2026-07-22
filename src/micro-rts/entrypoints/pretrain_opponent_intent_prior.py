"""Train a deployable history-conditioned multimodal opponent-intent prior."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from entrypoints.incomplete_info_common import (  # noqa: E402
    common_parser,
    load_config,
    load_full_state_tokenizer,
    load_stage_weights,
    make_loaders,
    measure_plan_stats,
    resolve_device,
    run_training,
    seed_all,
)
from models.incomplete_info import (  # noqa: E402
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    OpponentIntentPriorModel,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
    opponent_intent_prior_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/pretrain_opponent_intent_prior_medium_v2"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    model_cfg, training = cfg.model or {}, cfg.training or {}
    ego_cfg = EgoTokenizerConfig.from_dict(model_cfg.get("ego_tokenizer"))
    action_cfg = SelfActionTokenizerConfig.from_dict(
        model_cfg.get("self_action_tokenizer")
    )
    opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
        model_cfg.get("opponent_tokenizer")
    )
    history_cfg = HistoryConfig.from_dict(model_cfg.get("history"))
    intent_cfg = IntentPriorConfig.from_dict(model_cfg.get("intent_prior"))
    seq_len = int(training.get("seq_len", history_cfg.context_length))
    loader, val_loader, data = make_loaders(
        cfg, args, task="incomplete_dynamics", seq_len=seq_len
    )
    teacher, teacher_cfg, _ = load_full_state_tokenizer(
        training["full_state_tokenizer_ckpt"], loader.dataset, device
    )
    model = OpponentIntentPriorModel(
        teacher,
        loader.dataset.grid_hw,
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=intent_cfg,
    ).to(device)
    load_stage_weights(
        model.ego_tokenizer, training["ego_tokenizer_ckpt"], ("tokenizer.", "")
    )
    load_stage_weights(
        model.self_action_tokenizer, training["self_action_tokenizer_ckpt"]
    )
    load_stage_weights(
        model.opponent_tokenizer, training["opponent_tokenizer_ckpt"]
    )
    plan_mean, plan_std = measure_plan_stats(
        model.opponent_tokenizer,
        loader,
        device,
        batches=training.get("stats_batches", 32),
    )
    model.set_plan_stats(plan_mean, plan_std)
    model.freeze_teachers()
    coefficients = training.get("loss", {})

    def loss_fn(batch):
        return opponent_intent_prior_loss(
            model,
            batch,
            latent_coef=coefficients.get("latent", 1.0),
            event_coef=coefficients.get("event", 1.0),
            mode_coef=coefficients.get("mode", 0.25),
            balance_coef=coefficients.get("balance", 0.1),
            contrastive_coef=coefficients.get("contrastive", 0.5),
            shuffled_margin_coef=coefficients.get("shuffled_margin", 0.5),
            diversity_coef=coefficients.get("diversity", 0.05),
            shuffled_margin=training.get("shuffled_history_margin", 0.25),
            diversity_floor=training.get("diversity_floor", 0.75),
            contrastive_temperature=training.get("contrastive_temperature", 0.1),
        )

    return run_training(
        cfg,
        args,
        model,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="intent",
        metadata={
            "ego_tokenizer_cfg": ego_cfg.__dict__,
            "self_action_tokenizer_cfg": action_cfg.__dict__,
            "opponent_tokenizer_cfg": opponent_cfg.__dict__,
            "history_cfg": history_cfg.__dict__,
            "intent_prior_cfg": intent_cfg.__dict__,
            "tokenizer_cfg": teacher_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "full_state_tokenizer_ckpt": training["full_state_tokenizer_ckpt"],
            "ego_tokenizer_ckpt": training["ego_tokenizer_ckpt"],
            "self_action_tokenizer_ckpt": training["self_action_tokenizer_ckpt"],
            "opponent_tokenizer_ckpt": training["opponent_tokenizer_ckpt"],
            "data": str(data),
        },
        trainable_checkpoint_only=True,
    )


if __name__ == "__main__":
    main()
