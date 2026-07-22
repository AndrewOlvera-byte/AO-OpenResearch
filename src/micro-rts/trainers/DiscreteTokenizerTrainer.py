"""Registered trainer for the hard-code structured MicroRTS tokenizer."""

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
from core.registry import register  # noqa: E402
from entrypoints.pretrain_common import (  # noqa: E402
    amp_ctx,
    make_adam,
    make_lr_scheduler,
    pick,
    resolve_data,
    setup_backend,
)
from models.dreamer_v2 import (  # noqa: E402
    DiscreteStructuredTokenizer,
    DiscreteTokenizerConfig,
    discrete_reconstruction_loss,
)
from trainers.BaseTrainer import BaseTrainer, resolve_device  # noqa: E402
from trainers.PretrainCheckpointManager import PretrainCheckpointManager  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--exp",
        default="micro-rts/tokenizer/discrete_v3/pretrain_discrete_tokenizer_v3",
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


def _run(cfg, args):
    torch.manual_seed(int(cfg.run.get("seed", 0)))
    tr, mc = cfg.training or {}, cfg.model or {}
    steps = pick(args.steps, tr.get("steps"), 100000)
    batch = pick(args.batch, tr.get("batch"), 64)
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
        task="structured_tokenizer",
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
    ds = loader.dataset
    tc = DiscreteTokenizerConfig.from_dict(mc.get("tokenizer"))
    tc.mask_width = 1 + sum(ds.action_nvec[1:])
    tc.legacy_obs_channels = ds.obs_channels
    if args.smoke:
        tc.d_model, tc.depth, tc.n_heads = 32, 1, 4
        tc.codebook_size, tc.codebook_depth, tc.n_global_tokens = 16, 2, 2
    model = DiscreteStructuredTokenizer(ds.grid_hw, tc).to(device)
    lr = float(args.lr if args.lr is not None else tr.get("lr", 3e-4))
    opt = make_adam(
        model.parameters(), lr, device, weight_decay=float(tr.get("weight_decay", 1e-4))
    )
    sched = make_lr_scheduler(
        opt, steps, tr.get("warmup_steps", 1000), tr.get("lr_min_frac", 0.05)
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    out = args.out or tr.get("out", "checkpoints/discrete_tokenizer_v3.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "discrete_tokenizer_v3",
            "discrete_tokenizer_cfg": tc.__dict__,
            "grid_hw": ds.grid_hw,
            "code_shape": (model.n_tokens, model.codebook_depth),
        },
        default_monitor="val/dtok/exact_roundtrip",
    )
    start = checkpoints.load_if_requested(args.resume, args.resume_from)
    usage_coef = float(tr.get("usage_coef", 0.01))

    def loss_fn(batch_data):
        decoded, _codes, aux = model(batch_data["state"], batch_data["globals"])
        return discrete_reconstruction_loss(
            model,
            decoded,
            batch_data["state"],
            batch_data["globals"],
            aux["code_probs"],
            batch_data.get("obs"),
            batch_data.get("mask"),
            usage_coef=usage_coef,
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
                sums[f"val/{key}"] = sums.get(f"val/{key}", 0.0) + item
            count += 1
        values = {key: item / count for key, item in sums.items()}
        trainer.log(values, step=step)
        checkpoints.record_eval(step, values)
        model.train()
        print(
            f"[discrete-tokenizer] step={step} VAL "
            f"loss={values['val/dtok/total']:.4f} "
            f"cell={values['val/dtok/exact_cell']:.4f} "
            f"frame={values['val/dtok/exact_frame']:.4f} "
            f"raster={values['val/dtok/exact_raster']:.4f} "
            f"roundtrip={values['val/dtok/exact_roundtrip']:.4f}"
        )

    print(
        f"[discrete-tokenizer] data={data} device={device} "
        f"semantic={model.n_tokens} depth={model.codebook_depth} "
        f"base_codes={model.n_code_tokens} vocab={model.codebook_size}"
    )
    model.train()
    t0, last = time.time(), {}
    for step, batch_data in zip(range(start + 1, steps + 1), cycle(loader)):
        batch_data = to_device(batch_data, device)
        with amp_ctx(device, amp):
            loss, metrics = loss_fn(batch_data)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(tr.get("grad_clip", 5.0))
        )
        opt.step()
        sched.step()
        if step == 1 or step % int(tr.get("log_every", 100)) == 0 or step == steps:
            diag = {
                "dtok/grad_norm": float(grad),
                "dtok/lr": opt.param_groups[0]["lr"],
                "dtok/seq_per_s": step * batch / max(time.time() - t0, 1e-6),
            }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[discrete-tokenizer] step={step}/{steps} "
                f"loss={float(metrics['dtok/total']):.4f} "
                f"cell={float(metrics['dtok/exact_cell']):.4f} "
                f"frame={float(metrics['dtok/exact_frame']):.4f}"
            )
        if val_loader is not None and step % int(tr.get("eval_every", 1000)) == 0:
            run_val(step)
        last = metrics
        checkpoints.save_periodic(step, metrics)
    checkpoints.finish(steps, last, out)
    trainer.finish()
    print(f"[discrete-tokenizer] saved -> {out}")


@register("trainer", "discrete_v3_tokenizer")
class DiscreteTokenizerTrainer:
    def __init__(self, cfg, args, **_):
        self.cfg, self.args = cfg, args

    def train(self):
        return _run(self.cfg, self.args)

    smoke_test = train


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    trainer = DiscreteTokenizerTrainer(cfg, args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
