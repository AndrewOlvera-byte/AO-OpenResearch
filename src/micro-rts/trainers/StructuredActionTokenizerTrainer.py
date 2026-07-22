"""Registered trainer for the structured MicroRTS multi-event action tokenizer.

The state tokenizer is frozen.  The trainable action representation receives
dual forward/inverse self-supervision plus exact event-field reconstruction and
paired cloned-engine effect alignment.  Its checkpoint can be loaded and frozen
by ``train_dreamer_dynamics.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
for p in (HERE.parents[1], HERE.parents[2]):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

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
    ActionTokenizerConfig,
    ActionTokenizerPretrainer,
    StructuredTokenizer,
    StructuredTokenizerConfig,
    action_tokenizer_ssl_loss,
    structured_tokenizer_state_dict,
)
from trainers.BaseTrainer import BaseTrainer, resolve_device  # noqa: E402
from trainers.PretrainCheckpointManager import PretrainCheckpointManager  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--exp",
        default="micro-rts/tokenizer/structured_v2/pretrain_structured_action_tokenizer_v2",
    )
    p.add_argument("--data", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-key", default=None)
    p.add_argument("--resume", action="store_true", default=None)
    p.add_argument("--resume-from", default=None)
    p.add_argument("--set", action="append", default=[], metavar="K=V")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def _load_state_tokenizer(path, ds, device, stats_path=None):
    ckpt = torch.load(path, map_location="cpu")
    tc = StructuredTokenizerConfig.from_dict(ckpt["tokenizer_cfg"])
    tc.mask_width = 1 + sum(ds.action_nvec[1:])
    tc.legacy_obs_channels = ds.obs_channels
    tokenizer = StructuredTokenizer(ds.grid_hw, tc).to(device)
    tokenizer.load_state_dict(structured_tokenizer_state_dict(ckpt))
    tokenizer.requires_grad_(False)
    tokenizer.eval()
    stats_ckpt = ckpt
    if "latent_mean" not in stats_ckpt or "latent_std" not in stats_ckpt:
        if not stats_path:
            raise ValueError(
                f"{path}: structured tokenizer checkpoint lacks latent_mean/std; "
                "set training.tokenizer_stats_ckpt to a compatible final checkpoint"
            )
        stats_ckpt = torch.load(stats_path, map_location="cpu")
        stats_cfg = StructuredTokenizerConfig.from_dict(stats_ckpt["tokenizer_cfg"])
        if (stats_cfg.d_latent, stats_cfg.max_entities) != (
            tc.d_latent,
            tc.max_entities,
        ):
            raise ValueError(
                f"{stats_path}: latent geometry is incompatible with {path}"
            )
        if "latent_mean" not in stats_ckpt or "latent_std" not in stats_ckpt:
            raise ValueError(f"{stats_path}: checkpoint lacks latent_mean/std")
        ckpt = dict(ckpt)
        ckpt["latent_mean"] = stats_ckpt["latent_mean"]
        ckpt["latent_std"] = stats_ckpt["latent_std"]
    return tokenizer, tc, ckpt


def _run(cfg, args):
    tr, mc = cfg.training or {}, cfg.model or {}
    steps = pick(args.steps, tr.get("steps"), 80000)
    batch_size = pick(args.batch, tr.get("batch"), 32)
    workers = pick(args.num_workers, tr.get("num_workers"), 8)
    data = resolve_data(pick(args.data, (cfg.data or {}).get("path"), None))
    tokenizer_ckpt = tr.get("tokenizer_ckpt")
    if not tokenizer_ckpt:
        raise SystemExit("training.tokenizer_ckpt is required")
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    if args.smoke:
        steps, batch_size, workers, device = (
            min(steps, 2),
            min(batch_size, 2),
            0,
            torch.device("cpu"),
        )
    setup_backend(device)
    amp = bool(tr.get("amp", True)) and not args.smoke
    val_frac = 0.0 if args.smoke else float(tr.get("val_frac", 0.05))
    eval_every = int(tr.get("eval_every", 1000))
    eval_batches = int(tr.get("eval_batches", 16))
    fixed_val = bool(tr.get("fixed_val", True))
    common_loader = dict(
        path=data,
        task="structured_action_tokenizer",
        seq_len=1,
        batch_size=batch_size,
        locking=False,
        val_frac=val_frac,
    )
    loader = build_mrts_loader(
        **common_loader,
        num_workers=workers,
        split="train",
        paired_batch_fraction=tr.get("paired_batch_fraction"),
    )
    val_loader = (
        build_mrts_loader(
            **common_loader,
            num_workers=0,
            split="val",
            drop_last=False,
            shuffle=not fixed_val,
            fixed_chunk_batches=eval_batches if fixed_val else None,
            fixed_chunk_seed=int(tr.get("fixed_val_seed", 0)),
        )
        if val_frac
        else None
    )
    ds = loader.dataset
    state_tokenizer, tc, state_ckpt = _load_state_tokenizer(
        tokenizer_ckpt,
        ds,
        device,
        stats_path=tr.get("tokenizer_stats_ckpt"),
    )
    ac = ActionTokenizerConfig.from_dict(mc.get("action_tokenizer"))
    if args.smoke:
        ac.d_model = 64
        ac.field_dim = 8
        ac.n_heads = 4
        ac.inverse_depth = 1
        ac.max_action_events = min(ac.max_action_events, 8)
    model = ActionTokenizerPretrainer(
        state_tokenizer.n_tokens,
        state_tokenizer.d_latent,
        ds.grid_hw,
        ac,
    ).to(device)
    model.set_latent_stats(
        state_ckpt["latent_mean"].to(device),
        state_ckpt["latent_std"].to(device),
    )
    lr = float(args.lr if args.lr is not None else tr.get("lr", 2e-4))
    opt = make_adam(
        model.parameters(),
        lr,
        device,
        weight_decay=float(tr.get("weight_decay", 1e-4)),
    )
    sched = make_lr_scheduler(
        opt,
        steps,
        tr.get("warmup_steps", 1000),
        tr.get("lr_min_frac", 0.05),
    )
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()
    out = args.out or tr.get("out", "checkpoints/structured_action_tokenizer_v2.pt")
    checkpoints = PretrainCheckpointManager(
        trainer,
        model,
        opt,
        sched,
        metadata={
            "phase": "structured_action_tokenizer_v2",
            "action_tokenizer_cfg": ac.__dict__,
            "tokenizer_cfg": tc.__dict__,
            "grid_hw": ds.grid_hw,
            "action_nvec": ds.action_nvec,
            "state_tokenizer_ckpt": tokenizer_ckpt,
            "state_tokenizer_stats_ckpt": tr.get("tokenizer_stats_ckpt"),
        },
        default_monitor="val/action_tok/total",
    )
    start_step = checkpoints.load_if_requested(args.resume, args.resume_from)

    def loss_fn(b):
        return action_tokenizer_ssl_loss(
            model,
            state_tokenizer,
            b,
            reconstruction_coef=float(tr.get("reconstruction_coef", 1.0)),
            inverse_coef=float(tr.get("inverse_coef", 1.0)),
            forward_coef=float(tr.get("forward_coef", 1.0)),
            paired_effect_coef=float(tr.get("paired_effect_coef", 4.0)),
            alignment_coef=float(tr.get("alignment_coef", 0.1)),
            changed_token_boost=float(tr.get("changed_token_boost", 8.0)),
            change_threshold=float(tr.get("change_threshold", 1e-3)),
            padding_token_weight=float(tr.get("padding_token_weight", 0.05)),
        )

    @torch.no_grad()
    def run_val(step):
        model.eval()
        sums, n = {}, 0
        for vb in val_loader:
            if n >= eval_batches:
                break
            vb = to_device(vb, device)
            with amp_ctx(device, amp):
                _, vm = loss_fn(vb)
            for key, value in vm.items():
                name = f"val/{key}"
                sums[name] = sums.get(name, 0.0) + value
            n += 1
        vals = {key: value / n for key, value in sums.items()} if n else {}
        trainer.log(vals, step=step)
        checkpoints.record_eval(step, vals)
        model.train()
        if vals:
            print(
                f"[action-tokenizer] step={step} VAL "
                f"total={vals['val/action_tok/total']:.4f} "
                f"recon={vals['val/action_tok/reconstruction']:.4f} "
                f"inverse={vals['val/action_tok/inverse']:.4f} "
                f"forward={vals['val/action_tok/forward']:.4f} "
                f"paired={vals['val/action_tok/paired_effect']:.4f} "
                f"cos={vals['val/action_tok/effect_cosine']:.3f} "
                f"norm_agg="
                f"{vals['val/action_tok/effect_norm_ratio_aggregate']:.3f}"
            )

    print(
        f"[action-tokenizer] data={data} device={device} "
        f"amp={'bf16' if amp and device.type == 'cuda' else 'off'} "
        f"events={ac.max_action_events} width={ac.d_model}"
    )
    model.train()
    state_tokenizer.eval()
    log_every = int(tr.get("log_every", 100))
    grad_clip = float(tr.get("grad_clip", 5.0))
    t0 = time.time()
    last_metrics = {}
    for step, b in zip(range(start_step + 1, steps + 1), cycle(loader)):
        b = to_device(b, device)
        with amp_ctx(device, amp):
            loss, metrics = loss_fn(b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        if step == 1 or step % log_every == 0 or step == steps:
            diag = {
                "action_tok/grad_norm": float(grad),
                "action_tok/lr": opt.param_groups[0]["lr"],
                "action_tok/seq_per_s": step
                * batch_size
                / max(time.time() - t0, 1e-6),
            }
            trainer.log({**metrics, **diag}, step=step)
            print(
                f"[action-tokenizer] step={step}/{steps} "
                f"total={float(metrics['action_tok/total']):.4f} "
                f"recon={float(metrics['action_tok/reconstruction']):.4f} "
                f"inverse={float(metrics['action_tok/inverse']):.4f} "
                f"forward={float(metrics['action_tok/forward']):.4f} "
                f"paired={float(metrics['action_tok/paired_effect']):.4f} "
                f"cos={float(metrics['action_tok/effect_cosine']):.3f} "
                f"norm_agg="
                f"{float(metrics['action_tok/effect_norm_ratio_aggregate']):.3f} "
                f"({diag['action_tok/seq_per_s']:.0f} seq/s)",
                flush=True,
            )
        if val_loader is not None and step % eval_every == 0:
            run_val(step)
        last_metrics = metrics
        checkpoints.save_periodic(step, metrics)
    checkpoints.finish(steps, last_metrics, out)
    trainer.finish()
    print(f"[action-tokenizer] saved -> {out}")


@register("trainer", "structured_v2_action_tokenizer")
class StructuredActionTokenizerTrainer:
    """Config-selected facade owning the structured action-tokenizer run."""

    def __init__(self, cfg, args, **_):
        self.cfg = cfg
        self.args = args

    def train(self):
        return _run(self.cfg, self.args)

    def smoke_test(self):
        return _run(self.cfg, self.args)


def main(argv=None):
    args = parse_args(argv)
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    trainer = StructuredActionTokenizerTrainer(cfg, args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
