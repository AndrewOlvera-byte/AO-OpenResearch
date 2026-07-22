"""Pretrain the discrete action-JEPA and transferable next-code router."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
for path in (HERE.parents[1], HERE.parents[2]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from collectors.offline_data import build_mrts_loader, cycle, to_device  # noqa: E402
from core.config import Config  # noqa: E402
from entrypoints.pretrain_common import (  # noqa: E402
    amp_ctx,
    make_adam,
    make_lr_scheduler,
    pick,
    resolve_data,
    setup_backend,
)
from models.dreamer_v2 import (  # noqa: E402
    DiscreteActionTokenizerConfig,
    DiscreteActionTokenizerPretrainer,
    DiscreteStructuredTokenizer,
    DiscreteTokenizerConfig,
    discrete_action_jepa_loss,
)
from trainers.BaseTrainer import BaseTrainer, resolve_device  # noqa: E402
from trainers.PretrainCheckpointManager import PretrainCheckpointManager  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        default="micro-rts/tokenizer/discrete_v3/pretrain_discrete_action_tokenizer_v3",
    )
    parser.add_argument("--data", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-key", default=None)
    parser.add_argument("--resume", action="store_true", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _load_tokenizer(path, device):
    payload = torch.load(path, map_location="cpu")
    cfg = DiscreteTokenizerConfig.from_dict(payload["discrete_tokenizer_cfg"])
    model = DiscreteStructuredTokenizer(tuple(payload["grid_hw"]), cfg).to(device)
    model.load_state_dict(payload["model"])
    model.requires_grad_(False)
    model.eval()
    return model, cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    torch.manual_seed(int(cfg.run.get("seed", 0)))
    tr, mc = cfg.training or {}, cfg.model or {}
    steps = pick(args.steps, tr.get("steps"), 100000)
    batch = pick(args.batch, tr.get("batch"), 32)
    workers = pick(args.num_workers, tr.get("num_workers"), 8)
    data = resolve_data(pick(args.data, (cfg.data or {}).get("path"), None))
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    if args.smoke:
        steps, batch, workers, device = 2, min(batch, 2), 0, torch.device("cpu")
    setup_backend(device)
    amp = bool(tr.get("amp", True)) and not args.smoke
    val_frac = 0.0 if args.smoke else float(tr.get("val_frac", 0.05))
    common = dict(
        path=data,
        task="structured_action_tokenizer",
        seq_len=1,
        batch_size=batch,
        locking=False,
        val_frac=val_frac,
    )
    loader = build_mrts_loader(**common, num_workers=workers, split="train")
    val_loader = (
        build_mrts_loader(
            **common,
            num_workers=0,
            split="val",
            drop_last=False,
            shuffle=False,
            fixed_chunk_batches=int(tr.get("eval_batches", 16)),
            fixed_chunk_seed=int(tr.get("fixed_val_seed", 0)),
        )
        if val_frac
        else None
    )
    tokenizer_path = tr.get("tokenizer_ckpt")
    if not tokenizer_path:
        raise SystemExit("training.tokenizer_ckpt is required")
    tokenizer, tokenizer_cfg = _load_tokenizer(tokenizer_path, device)
    action_cfg = DiscreteActionTokenizerConfig.from_dict(mc.get("action_tokenizer"))
    if args.smoke:
        action_cfg.d_model, action_cfg.field_dim = 64, 8
        action_cfg.n_heads, action_cfg.inverse_depth = 4, 1
        action_cfg.max_action_events = 8
    model = DiscreteActionTokenizerPretrainer(
        tokenizer,
        tokenizer.grid_hw if hasattr(tokenizer, "grid_hw") else loader.dataset.grid_hw,
        action_cfg,
    ).to(device)
    lr = float(args.lr if args.lr is not None else tr.get("lr", 2e-4))
    opt = make_adam(
        model.parameters(),
        lr,
        device,
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )
    sched = make_lr_scheduler(
        opt, steps, tr.get("warmup_steps", 1000), tr.get("lr_min_frac", 0.05)
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    out = args.out or tr.get("out", "checkpoints/discrete_action_tokenizer_v3.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "discrete_action_tokenizer_v3",
            "discrete_action_tokenizer_cfg": action_cfg.__dict__,
            "discrete_tokenizer_cfg": tokenizer_cfg.__dict__,
            "grid_hw": loader.dataset.grid_hw,
            "state_tokenizer_ckpt": tokenizer_path,
        },
        default_monitor="val/dajepa/total",
    )
    start = checkpoints.load_if_requested(args.resume, args.resume_from)

    def loss_fn(value):
        return discrete_action_jepa_loss(
            model,
            tokenizer,
            value,
            reconstruction_coef=float(tr.get("reconstruction_coef", 1.0)),
            inverse_coef=float(tr.get("inverse_coef", 1.0)),
            forward_coef=float(tr.get("forward_coef", 1.0)),
            counterfactual_coef=float(tr.get("counterfactual_coef", 1.0)),
            effect_margin_coef=float(tr.get("effect_margin_coef", 2.0)),
            alignment_coef=float(tr.get("alignment_coef", 0.1)),
            effect_margin=float(tr.get("effect_margin", 1.0)),
        )

    @torch.no_grad()
    def run_val(step):
        model.eval()
        sums, count = {}, 0
        for value in val_loader:
            value = to_device(value, device)
            with amp_ctx(device, amp):
                _, metrics = loss_fn(value)
            for key, item in metrics.items():
                name = f"val/{key}"
                sums[name] = sums.get(name, 0.0) + item
            count += 1
        values = {key: item / count for key, item in sums.items()}
        trainer.log(values, step=step)
        checkpoints.record_eval(step, values)
        model.train()
        print(
            f"[discrete-action-jepa] step={step} VAL "
            f"total={values['val/dajepa/total']:.4f} "
            f"code={values['val/dajepa/code_acc']:.4f} "
            f"changed={values['val/dajepa/changed_code_acc']:.4f}"
        )

    print(
        f"[discrete-action-jepa] data={data} device={device} "
        f"events={action_cfg.max_action_events} state_codes={tokenizer.n_code_tokens}"
    )
    model.train()
    tokenizer.eval()
    t0, last = time.time(), {}
    for step, value in zip(range(start + 1, steps + 1), cycle(loader)):
        value = to_device(value, device)
        with amp_ctx(device, amp):
            loss, metrics = loss_fn(value)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tr.get("grad_clip", 5.0))
        )
        opt.step()
        sched.step()
        if step == 1 or step % int(tr.get("log_every", 100)) == 0 or step == steps:
            diag = {
                "dajepa/grad_norm": float(grad),
                "dajepa/lr": opt.param_groups[0]["lr"],
                "dajepa/seq_per_s": step * batch / max(time.time() - t0, 1e-6),
            }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[discrete-action-jepa] step={step}/{steps} "
                f"total={float(metrics['dajepa/total']):.4f} "
                f"code={float(metrics['dajepa/code_acc']):.4f} "
                f"changed={float(metrics['dajepa/changed_code_acc']):.4f}"
            )
        if val_loader is not None and step % int(tr.get("eval_every", 1000)) == 0:
            run_val(step)
        last = metrics
        checkpoints.save_periodic(step, metrics)
    checkpoints.finish(steps, last, out)
    trainer.finish()
    print(f"[discrete-action-jepa] saved -> {out}")


if __name__ == "__main__":
    main()
