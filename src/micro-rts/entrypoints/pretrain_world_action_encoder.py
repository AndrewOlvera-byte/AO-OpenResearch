"""Pretrain the transition-centric predictive belief encoder for CausalWorldAction."""

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
    EgoObservationTokenizer,
    EgoTokenizerConfig,
    OpponentPlanTokenizer,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizer,
    SelfActionTokenizerConfig,
)
from models.incomplete_info.world_action import (  # noqa: E402
    PredictiveBeliefConfig,
    PredictiveBeliefPretrainer,
    predictive_belief_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__,
        "micro-rts/paper/incomplete_info/causal_world_action_v1/"
        "pretrain_predictive_belief_encoder",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    model_cfg, training = cfg.model or {}, cfg.training or {}
    belief_cfg = PredictiveBeliefConfig.from_dict(model_cfg.get("predictive_belief"))

    source_path = resolve_path(training["source_intent_ckpt"])
    source = torch.load(source_path, map_location="cpu", weights_only=False)
    seq_len = int(training.get("seq_len", belief_cfg.context_length))
    loader, val_loader, data = make_loaders(
        cfg,
        args,
        task="incomplete_dynamics_paired",
        seq_len=seq_len,
    )
    full_tokenizer, tokenizer_cfg, _ = load_full_state_tokenizer(
        training.get("full_state_tokenizer_ckpt", source["full_state_tokenizer_ckpt"]),
        loader.dataset,
        device,
    )
    ego_cfg = EgoTokenizerConfig.from_dict(source["ego_tokenizer_cfg"])
    action_cfg = SelfActionTokenizerConfig.from_dict(source["self_action_tokenizer_cfg"])
    opponent_cfg = OpponentPlanTokenizerConfig.from_dict(
        source["opponent_tokenizer_cfg"]
    )
    ego = EgoObservationTokenizer(loader.dataset.grid_hw, ego_cfg).to(device)
    action = SelfActionTokenizer(
        full_tokenizer.n_tokens, loader.dataset.grid_hw, action_cfg
    ).to(device)
    opponent = OpponentPlanTokenizer(
        full_tokenizer.n_tokens, loader.dataset.grid_hw, opponent_cfg
    ).to(device)
    load_stage_weights(
        ego,
        training.get("ego_tokenizer_ckpt", source["ego_tokenizer_ckpt"]),
        ("tokenizer.", ""),
    )
    load_stage_weights(
        action,
        training.get("self_action_tokenizer_ckpt", source["self_action_tokenizer_ckpt"]),
    )
    load_stage_weights(
        opponent,
        training.get("opponent_tokenizer_ckpt", source["opponent_tokenizer_ckpt"]),
    )
    model = PredictiveBeliefPretrainer(ego, action, opponent, belief_cfg).to(device)
    coefficients = training.get("loss", {})

    def loss_fn(batch):
        return predictive_belief_loss(
            model,
            batch,
            jepa_coef=coefficients.get("future_jepa", 1.0),
            variance_coef=coefficients.get("variance", 0.05),
            inverse_coef=coefficients.get("self_inverse", 0.5),
            opponent_coef=coefficients.get("opponent_plan", 0.5),
            event_coef=coefficients.get("events", 0.5),
            reward_coef=coefficients.get("reward", 0.5),
            return_coef=coefficients.get("return", 0.5),
            continue_coef=coefficients.get("continue", 0.1),
            counterfactual_coef=coefficients.get("counterfactual", 1.0),
            counterfactual_effect_coef=coefficients.get(
                "counterfactual_effect", 1.0
            ),
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    resident = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[causal-world-action-encoder] source={source_path} "
        f"trainable={trainable:,} resident={resident:,} "
        f"amp={'bf16' if training.get('amp', True) else 'off'} "
        "attention=sdpa-flash temporal=causal",
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
        phase="world_action_encoder",
        metadata={
            "predictive_belief_cfg": belief_cfg.__dict__,
            "tokenizer_cfg": tokenizer_cfg.__dict__,
            "ego_tokenizer_cfg": ego_cfg.__dict__,
            "self_action_tokenizer_cfg": action_cfg.__dict__,
            "opponent_tokenizer_cfg": opponent_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "source_intent_ckpt": str(source_path),
            "full_state_tokenizer_ckpt": training.get(
                "full_state_tokenizer_ckpt", source["full_state_tokenizer_ckpt"]
            ),
            "ego_tokenizer_ckpt": training.get(
                "ego_tokenizer_ckpt", source["ego_tokenizer_ckpt"]
            ),
            "self_action_tokenizer_ckpt": training.get(
                "self_action_tokenizer_ckpt", source["self_action_tokenizer_ckpt"]
            ),
            "opponent_tokenizer_ckpt": training.get(
                "opponent_tokenizer_ckpt", source["opponent_tokenizer_ckpt"]
            ),
            "data": str(data),
            "architecture": "CausalWorldAction-v1",
        },
        after_step=model.update_target,
        trainable_checkpoint_only=True,
        checkpoint_include_prefixes=("target_encoder.",),
    )


if __name__ == "__main__":
    main()
