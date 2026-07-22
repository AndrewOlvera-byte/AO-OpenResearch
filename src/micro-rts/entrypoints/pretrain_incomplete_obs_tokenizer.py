"""Pretrain the fog-aware ego observation tokenizer with masked JEPA targets."""

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
    EgoTokenizerConfig,
    EgoTokenizerPretrainer,
    ego_tokenizer_loss,
)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/probe_ego_tokenizer"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    loader, val_loader, data = make_loaders(
        cfg, args, task="incomplete_obs_tokenizer", seq_len=1
    )
    teacher, teacher_cfg, _ = load_full_state_tokenizer(
        cfg.training["full_state_tokenizer_ckpt"],
        loader.dataset,
        device,
        cfg.training.get("full_state_tokenizer_stats_ckpt"),
    )
    model_cfg = EgoTokenizerConfig.from_dict((cfg.model or {}).get("ego_tokenizer"))
    if model_cfg.d_latent != teacher.d_latent:
        raise ValueError("ego and full-state tokenizer latent dimensions must match")
    model = EgoTokenizerPretrainer(loader.dataset.grid_hw, model_cfg).to(device)
    coefficients = (cfg.training or {}).get("loss", {})

    def loss_fn(batch):
        return ego_tokenizer_loss(
            model,
            batch,
            full_state_tokenizer=teacher,
            reconstruction_coef=coefficients.get("reconstruction", 1.0),
            visibility_coef=coefficients.get("visibility", 0.1),
            jepa_coef=coefficients.get("jepa", 0.25),
            teacher_coef=coefficients.get("full_teacher", 0.25),
        )

    return run_training(
        cfg,
        args,
        model,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="ego",
        metadata={
            "ego_tokenizer_cfg": model_cfg.__dict__,
            "tokenizer_cfg": teacher_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "data": str(data),
        },
        after_step=model.update_target,
    )


if __name__ == "__main__":
    main()
