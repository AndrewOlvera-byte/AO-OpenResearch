"""Pretrain deployable self-action events from ego-local observations."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch  # noqa: E402

from entrypoints.incomplete_info_common import (  # noqa: E402
    common_parser,
    load_config,
    load_full_state_tokenizer,
    make_loaders,
    resolve_device,
    resolve_path,
    run_training,
    seed_all,
)
from models.incomplete_info import (  # noqa: E402
    SelfActionTokenizer,
    SelfActionTokenizerConfig,
    self_action_tokenizer_loss,
)


def _initialize_event_encoder(model, path):
    checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    source = checkpoint.get("model", checkpoint)
    mapped = {
        key.removeprefix("action_encoder."): value
        for key, value in source.items()
        if key.startswith("action_encoder.")
    }
    if mapped:
        model.event_encoder.load_state_dict(mapped)


def main(argv=None):
    parser = common_parser(
        __doc__, "micro-rts/paper/incomplete_info/probe_self_action_tokenizer"
    )
    args = parser.parse_args(argv)
    cfg = load_config(args)
    seed_all(int((cfg.run or {}).get("seed", 0)))
    device = resolve_device(args.device or (cfg.run or {}).get("device"))
    loader, val_loader, data = make_loaders(
        cfg, args, task="incomplete_action_tokenizer", seq_len=1
    )
    teacher, teacher_cfg, _ = load_full_state_tokenizer(
        cfg.training["full_state_tokenizer_ckpt"],
        loader.dataset,
        device,
        cfg.training.get("full_state_tokenizer_stats_ckpt"),
    )
    model_cfg = SelfActionTokenizerConfig.from_dict(
        (cfg.model or {}).get("self_action_tokenizer")
    )
    if model_cfg.d_latent != teacher.d_latent:
        raise ValueError("action and state latent dimensions must match")
    model = SelfActionTokenizer(teacher.n_tokens, loader.dataset.grid_hw, model_cfg).to(
        device
    )
    if cfg.training.get("init_action_tokenizer_ckpt"):
        _initialize_event_encoder(model, cfg.training["init_action_tokenizer_ckpt"])
    coefficients = cfg.training.get("loss", {})

    def loss_fn(batch):
        return self_action_tokenizer_loss(
            model,
            teacher,
            batch,
            reconstruction_coef=coefficients.get("reconstruction", 1.0),
            forward_coef=coefficients.get("forward", 1.0),
            changed_token_boost=coefficients.get("changed_token_boost", 8.0),
        )

    return run_training(
        cfg,
        args,
        model,
        loss_fn,
        loader,
        val_loader,
        device,
        phase="self_action",
        metadata={
            "self_action_tokenizer_cfg": model_cfg.__dict__,
            "tokenizer_cfg": teacher_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "data": str(data),
        },
    )


if __name__ == "__main__":
    main()
