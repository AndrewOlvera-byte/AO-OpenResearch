"""Train state-only incomplete-information dynamics behind a frozen intent prior."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

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
    resolve_device,
    resolve_path,
    run_training,
    seed_all,
)
from models.incomplete_info import (  # noqa: E402
    BeliefDynamicsConfig,
    BeliefDynamicsModel,
    EgoTokenizerConfig,
    HistoryConfig,
    IntentPriorConfig,
    OpponentIntentPriorModel,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
    belief_dynamics_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/pretrain_belief_dynamics_medium_v1"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    model_cfg, training = cfg.model or {}, cfg.training or {}
    flow_cfg = BeliefDynamicsConfig.from_dict(model_cfg.get("belief_dynamics"))

    intent_path = resolve_path(training["intent_prior_ckpt"])
    intent_checkpoint = torch.load(intent_path, map_location="cpu", weights_only=False)
    history_cfg = HistoryConfig.from_dict(intent_checkpoint["history_cfg"])
    opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
        intent_checkpoint["opponent_tokenizer_cfg"]
    )
    seq_len = int(training.get("seq_len", history_cfg.context_length))
    paired = float(training.get("loss", {}).get("counterfactual", 0.0)) > 0.0
    loader, val_loader, data = make_loaders(
        cfg,
        args,
        task=("incomplete_dynamics_paired" if paired else "incomplete_dynamics"),
        seq_len=seq_len,
    )

    full_tokenizer_path = training.get(
        "full_state_tokenizer_ckpt",
        intent_checkpoint["full_state_tokenizer_ckpt"],
    )
    teacher, tokenizer_cfg, _ = load_full_state_tokenizer(
        full_tokenizer_path, loader.dataset, device
    )
    ego_cfg = EgoTokenizerConfig.from_dict(intent_checkpoint["ego_tokenizer_cfg"])
    action_cfg = SelfActionTokenizerConfig.from_dict(
        intent_checkpoint["self_action_tokenizer_cfg"]
    )
    intent_cfg = IntentPriorConfig.from_dict(intent_checkpoint["intent_prior_cfg"])
    intent_model = OpponentIntentPriorModel(
        teacher,
        loader.dataset.grid_hw,
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        intent_cfg=intent_cfg,
    ).to(device)
    load_stage_weights(
        intent_model.ego_tokenizer,
        training.get("ego_tokenizer_ckpt", intent_checkpoint["ego_tokenizer_ckpt"]),
        ("tokenizer.", ""),
    )
    load_stage_weights(
        intent_model.self_action_tokenizer,
        training.get(
            "self_action_tokenizer_ckpt",
            intent_checkpoint["self_action_tokenizer_ckpt"],
        ),
    )
    load_stage_weights(
        intent_model.opponent_tokenizer,
        training.get(
            "opponent_tokenizer_ckpt",
            intent_checkpoint["opponent_tokenizer_ckpt"],
        ),
    )
    load_stage_weights(intent_model, intent_path)

    model = BeliefDynamicsModel(intent_model, flow_cfg).to(device)
    mechanics_stats_path = resolve_path(training["mechanics_stats_ckpt"])
    mechanics_checkpoint = torch.load(
        mechanics_stats_path, map_location="cpu", weights_only=False
    )
    state_mean = mechanics_checkpoint["model"].get("dynamics.latent_mean")
    state_std = mechanics_checkpoint["model"].get("dynamics.latent_std")
    if state_mean is None or state_std is None:
        raise ValueError("mechanics checkpoint lacks dynamics latent statistics")
    model.set_state_stats(state_mean.to(device), state_std.to(device))
    model.freeze_conditioner()
    init_from = training.get("init_from")
    if init_from:
        init_path = resolve_path(init_from)
        initial = torch.load(init_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(initial["model"], strict=False)
        allowed_new = (
            "flow.action_residual_"
            if flow_cfg.explicit_action_residual
            else "__no_new_flow_parameters__"
        )
        bad_missing = [
            name
            for name in missing
            if not name.startswith("intent_model.")
            and not name.startswith(allowed_new)
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
    trainable_scope = training.get("trainable_scope", "flow")
    if trainable_scope == "action_residual":
        if not model.flow.explicit_action_residual:
            raise ValueError(
                "trainable_scope=action_residual requires "
                "model.belief_dynamics.explicit_action_residual=true"
            )
        model.flow.requires_grad_(False)
        for parameter in model.flow.action_residual_parameters():
            parameter.requires_grad_(True)
    elif trainable_scope != "flow":
        raise ValueError(f"unknown belief dynamics trainable_scope {trainable_scope!r}")
    coefficients = training.get("loss", {})

    def loss_fn(batch):
        return belief_dynamics_loss(
            model,
            batch,
            flow_coef=coefficients.get("flow", 1.0),
            prior_coef=coefficients.get("prior", 1.0),
            grounding_coef=coefficients.get("grounding", 0.25),
            history_rank_coef=coefficients.get("history_rank", 1.0),
            intent_rank_coef=coefficients.get("intent_rank", 1.0),
            action_rank_coef=coefficients.get("action_rank", 0.0),
            condition_margin=training.get("condition_margin", 0.05),
            visible_boost=training.get("visible_boost", 3.0),
            occupied_boost=training.get("occupied_boost", 0.0),
            hidden_occupied_boost=training.get("hidden_occupied_boost", 0.0),
            rank_anchors=training.get("rank_anchors", 1),
            action_residual_coef=coefficients.get("action_residual", 0.0),
            anchor_coef=coefficients.get("anchor", 0.0),
            anchor_grounding_coef=coefficients.get("anchor_grounding", 0.0),
            counterfactual_coef=coefficients.get("counterfactual", 0.0),
            counterfactual_effect_coef=coefficients.get(
                "counterfactual_effect", 0.0
            ),
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    resident = sum(p.numel() for p in model.parameters())
    print(
        f"[belief-dynamics] intent={intent_path} state_target=t+1 "
        f"trainable={trainable:,} resident={resident:,}",
        flush=True,
    )
    return run_training(
        cfg,
        args,
        model,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="belief_dynamics",
        metadata={
            "belief_dynamics_cfg": flow_cfg.__dict__,
            "tokenizer_cfg": tokenizer_cfg.__dict__,
            "ego_tokenizer_cfg": ego_cfg.__dict__,
            "self_action_tokenizer_cfg": action_cfg.__dict__,
            "opponent_tokenizer_cfg": opponent_cfg.__dict__,
            "history_cfg": history_cfg.__dict__,
            "intent_prior_cfg": intent_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "intent_prior_ckpt": str(intent_path),
            "mechanics_stats_ckpt": str(mechanics_stats_path),
            "full_state_tokenizer_ckpt": str(full_tokenizer_path),
            "data": str(data),
            "trainable_scope": trainable_scope,
        },
        trainable_checkpoint_only=True,
        checkpoint_include_prefixes=(
            ("flow.",) if trainable_scope == "action_residual" else ()
        ),
    )


if __name__ == "__main__":
    main()
