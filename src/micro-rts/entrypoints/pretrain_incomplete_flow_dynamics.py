"""Train causal history conditioning and joint hidden-state/opponent-plan flow."""

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
    load_frozen_mechanics,
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
    IncompleteInformationWorldModel,
    JointFlowConfig,
    OpponentPlanTokenizerConfig,
    SelfActionTokenizerConfig,
    joint_flow_world_model_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/probe_joint_flow_dynamics"
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
    flow_cfg = JointFlowConfig.from_dict(model_cfg.get("flow"))
    seq_len = int(
        training.get(
            "seq_len",
            max(history_cfg.context_length, max(opponent_cfg.horizons) + 1),
        )
    )
    loader, val_loader, data = make_loaders(
        cfg, args, task="incomplete_dynamics", seq_len=seq_len
    )
    mechanics, mechanics_checkpoint = load_frozen_mechanics(
        training["mechanics_ckpt"], device
    )
    world = IncompleteInformationWorldModel(
        mechanics.tokenizer,
        loader.dataset.grid_hw,
        ego_cfg=ego_cfg,
        self_action_cfg=action_cfg,
        opponent_cfg=opponent_cfg,
        history_cfg=history_cfg,
        flow_cfg=flow_cfg,
        mechanics=mechanics,
    ).to(device)
    load_stage_weights(
        world.ego_tokenizer, training["ego_tokenizer_ckpt"], ("tokenizer.", "")
    )
    load_stage_weights(
        world.self_action_tokenizer, training["self_action_tokenizer_ckpt"]
    )
    load_stage_weights(world.opponent_tokenizer, training["opponent_tokenizer_ckpt"])
    state_mean = mechanics_checkpoint["model"].get("dynamics.latent_mean")
    state_std = mechanics_checkpoint["model"].get("dynamics.latent_std")
    if state_mean is None or state_std is None:
        raise ValueError("mechanics checkpoint lacks dynamics latent statistics")
    plan_mean, plan_std = measure_plan_stats(
        world.opponent_tokenizer,
        loader,
        device,
        batches=training.get("stats_batches", 16),
    )
    world.set_latent_stats(
        state_mean.to(device), state_std.to(device), plan_mean, plan_std
    )
    world.freeze_teachers()
    coefficients = training.get("loss", {})

    def loss_fn(batch):
        return joint_flow_world_model_loss(
            world,
            batch,
            flow_coef=coefficients.get("flow", 1.0),
            grounding_coef=coefficients.get("grounding", 0.25),
            opponent_event_coef=coefficients.get("opponent_event", 0.5),
            future_jepa_coef=coefficients.get("future_jepa", 0.25),
        )

    return run_training(
        cfg,
        args,
        world,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="joint",
        metadata={
            "ego_tokenizer_cfg": ego_cfg.__dict__,
            "self_action_tokenizer_cfg": action_cfg.__dict__,
            "opponent_tokenizer_cfg": opponent_cfg.__dict__,
            "history_cfg": history_cfg.__dict__,
            "flow_cfg": flow_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "mechanics_ckpt": training["mechanics_ckpt"],
            "ego_tokenizer_ckpt": training["ego_tokenizer_ckpt"],
            "self_action_tokenizer_ckpt": training["self_action_tokenizer_ckpt"],
            "opponent_tokenizer_ckpt": training["opponent_tokenizer_ckpt"],
            "data": str(data),
        },
        trainable_checkpoint_only=True,
    )


if __name__ == "__main__":
    main()
