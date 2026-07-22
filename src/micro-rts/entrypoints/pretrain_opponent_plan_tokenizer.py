"""Pretrain the privileged multi-horizon opponent plan tokenizer."""

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
    make_loaders,
    resolve_device,
    run_training,
    seed_all,
)
from models.incomplete_info import (  # noqa: E402
    OpponentPlanTokenizer,
    OpponentPlanTokenizerConfig,
    opponent_plan_tokenizer_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/probe_opponent_plan_tokenizer"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    model_cfg = OpponentPlanTokenizerConfig.from_dict(
        (cfg.model or {}).get("opponent_tokenizer")
    )
    seq_len = int(cfg.training.get("seq_len", max(model_cfg.horizons) + 1))
    loader, val_loader, data = make_loaders(
        cfg, args, task="incomplete_opponent_tokenizer", seq_len=seq_len
    )
    teacher, teacher_cfg, _ = load_full_state_tokenizer(
        cfg.training["full_state_tokenizer_ckpt"],
        loader.dataset,
        device,
        cfg.training.get("full_state_tokenizer_stats_ckpt"),
    )
    if model_cfg.d_latent != teacher.d_latent:
        raise ValueError("opponent plan and state latent dimensions must match")
    model = OpponentPlanTokenizer(
        teacher.n_tokens, loader.dataset.grid_hw, model_cfg
    ).to(device)
    coefficients = cfg.training.get("loss", {})

    def loss_fn(batch):
        return opponent_plan_tokenizer_loss(
            model,
            teacher,
            batch,
            event_coef=coefficients.get("event", 1.0),
            future_state_coef=coefficients.get("future_state", 0.5),
        )

    return run_training(
        cfg,
        args,
        model,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="opponent_plan",
        metadata={
            "opponent_tokenizer_cfg": model_cfg.__dict__,
            "tokenizer_cfg": teacher_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "data": str(data),
        },
    )


if __name__ == "__main__":
    main()
