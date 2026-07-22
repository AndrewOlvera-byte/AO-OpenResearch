from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from collectors.offline_data import build_mrts_loader, cycle, to_device
from core.config import Config
from entrypoints.pretrain_common import (
    amp_ctx,
    make_adam,
    make_lr_scheduler,
    setup_backend,
)
from models.dreamer_v2 import (
    StructuredDynamicsConfig,
    StructuredTokenizer,
    StructuredTokenizerConfig,
    StructuredWorldModelV2,
    structured_tokenizer_state_dict,
)
from trainers.BaseTrainer import BaseTrainer


def common_parser(description, default_exp):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--exp", default=default_exp)
    parser.add_argument("--data")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-key")
    parser.add_argument("--set", action="append", default=[], metavar="K=V")
    return parser


def load_config(args):
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    return cfg


def resolve_path(pattern):
    paths = [Path(path) for path in glob.glob(str(pattern))]
    if paths:
        return max(paths, key=lambda path: path.stat().st_mtime)
    path = Path(pattern)
    if not path.exists():
        raise FileNotFoundError(f"no file matches {pattern!r}")
    return path


def resolve_device(value):
    value = value or "auto"
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(cfg, args, *, task, seq_len):
    training, data_cfg = cfg.training or {}, cfg.data or {}
    data = resolve_path(args.data or data_cfg["path"])
    batch = int(args.batch or training.get("batch", 16))
    workers = int(
        args.num_workers
        if args.num_workers is not None
        else training.get("num_workers", 4)
    )
    val_frac = float(training.get("val_frac", 0.05))
    if args.smoke:
        batch, workers, val_frac = min(batch, 2), 0, 0.0
    common = dict(
        path=str(data),
        task=task,
        seq_len=int(seq_len),
        batch_size=batch,
        num_workers=workers,
        locking=False,
        val_frac=val_frac,
        observation_mode=data_cfg.get("observation_mode", "ego"),
        pin_memory=bool(training.get("pin_memory", True)),
        prefetch_factor=int(training.get("prefetch_factor", 3)),
        persistent_workers=bool(training.get("persistent_workers", workers > 0)),
        h5_cache_mb=int(data_cfg.get("h5_cache_mb", 128)),
    )
    # Paired sampling is a property of the counterfactual training task, not
    # of every loader built from a config that happens to enable it.  Promotion
    # evaluation intentionally requests the ordinary dynamics schema, which
    # does not expose ``counterfactual_valid`` to its sampler.
    paired_fraction = (
        data_cfg.get("paired_batch_fraction")
        if task == "incomplete_dynamics_paired"
        else None
    )
    train = build_mrts_loader(
        **common,
        split="train",
        shuffle=True,
        paired_batch_fraction=paired_fraction,
    )
    val = None
    if val_frac:
        val_common = dict(common)
        val_common.update(num_workers=0, drop_last=False, shuffle=False)
        val = build_mrts_loader(**val_common, split="val")
    return train, val, data


def load_full_state_tokenizer(path, dataset, device, stats_path=None):
    checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    tokenizer_cfg = StructuredTokenizerConfig.from_dict(checkpoint["tokenizer_cfg"])
    tokenizer_cfg.mask_width = 1 + sum(dataset.action_nvec[1:])
    tokenizer_cfg.legacy_obs_channels = dataset.obs_channels
    tokenizer = StructuredTokenizer(dataset.grid_hw, tokenizer_cfg).to(device)
    tokenizer.load_state_dict(structured_tokenizer_state_dict(checkpoint))
    tokenizer.requires_grad_(False).eval()
    stats = checkpoint
    if ("latent_mean" not in stats or "latent_std" not in stats) and stats_path:
        stats = torch.load(
            resolve_path(stats_path), map_location="cpu", weights_only=False
        )
    return tokenizer, tokenizer_cfg, stats


def load_frozen_mechanics(path, device):
    checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    tokenizer_cfg = StructuredTokenizerConfig.from_dict(checkpoint["tokenizer_cfg"])
    dynamics_cfg = StructuredDynamicsConfig.from_dict(checkpoint["dynamics_cfg"])
    model = StructuredWorldModelV2(
        checkpoint.get("grid_hw", (16, 16)), tokenizer_cfg, dynamics_cfg
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.requires_grad_(False).eval()
    return model, checkpoint


def load_stage_weights(module, path, prefixes=("",)):
    checkpoint = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if candidate:
            missing, unexpected = module.load_state_dict(candidate, strict=False)
            if not unexpected and len(missing) < len(module.state_dict()):
                return checkpoint, missing
    raise ValueError(f"{path}: no compatible weights for {type(module).__name__}")


@torch.no_grad()
def measure_plan_stats(model, loader, device, batches=16):
    total = total2 = None
    count = 0
    for index, batch in enumerate(loader):
        if index >= int(batches):
            break
        batch = to_device(batch, device)
        plan = model.encode(batch["state"], batch["opponent_action"])[0].float()
        dims = tuple(range(plan.ndim - 1))
        total = plan.sum(dims) if total is None else total + plan.sum(dims)
        total2 = (
            plan.square().sum(dims)
            if total2 is None
            else total2 + plan.square().sum(dims)
        )
        count += plan.numel() // plan.shape[-1]
    mean = total / max(count, 1)
    std = (total2 / max(count, 1) - mean.square()).clamp_min(1e-6).sqrt()
    return mean, std


def run_training(
    cfg,
    args,
    model,
    loss_fn,
    train_loader,
    val_loader,
    device,
    *,
    phase,
    metadata,
    after_step=None,
    trainable_checkpoint_only=False,
    checkpoint_include_prefixes=(),
):
    training = cfg.training or {}
    setup_backend(device)
    steps = int(args.steps or training.get("steps", 10000))
    if args.smoke:
        steps = min(steps, 2)
    lr = float(training.get("lr", 2e-4))
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = make_adam(
        parameters,
        lr,
        device,
        eps=float(training.get("adam_eps", 1e-5)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scheduler = make_lr_scheduler(
        optimizer,
        steps,
        int(training.get("warmup_steps", 0)),
        float(training.get("lr_min_frac", 0.05)),
        training.get("lr_stages"),
    )
    out = Path(args.out or training.get("out", f"checkpoints/{phase}/model.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    start = 0
    if args.resume and out.exists():
        resume = torch.load(out, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=not trainable_checkpoint_only)
        optimizer.load_state_dict(resume["optimizer"])
        if "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
        start = int(resume["step"])
    amp = bool(training.get("amp", True)) and device.type == "cuda" and not args.smoke
    log_every = max(1, int(training.get("log_every", 100)))
    eval_every = max(1, int(training.get("eval_every", 1000)))
    eval_batches = max(1, int(training.get("eval_batches", 8)))
    eval_seed = training.get("eval_seed")
    checkpoint_every = max(1, int(training.get("checkpoint_every", 5000)))
    grad_clip = float(training.get("grad_clip", 5.0))
    log_metrics = training.get("log_metrics")
    best = float("inf")

    def metrics_for_logging(metrics):
        """Keep W&B/console readable when a loss emits deep diagnostic trees."""
        if not log_metrics:
            return dict(metrics)
        return {key: metrics[key] for key in log_metrics if key in metrics}

    # Keep these lightweight entrypoints aligned with the established
    # tokenizer/dynamics trainers: config-driven W&B, cached-container login,
    # and explicit CLI opt-out.  Use the run name when a config does not repeat
    # it under ``wandb.run_name``.
    wandb_cfg = cfg.wandb or {}
    if not wandb_cfg.get("run_name"):
        wandb_cfg["run_name"] = (cfg.run or {}).get("name", phase)
    trainer = BaseTrainer(cfg, device=str(device))
    trainer.use_wandb = not (args.no_wandb or args.smoke)
    trainer._wandb_key = args.wandb_key
    trainer.init_wandb()

    def save(path, step, metrics):
        state = model.state_dict()
        if trainable_checkpoint_only:
            trainable_names = {
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            root_buffers = {"state_mean", "state_std", "plan_mean", "plan_std"}
            state = {
                name: value
                for name, value in state.items()
                if name in trainable_names
                or name in root_buffers
                or any(
                    name.startswith(prefix)
                    for prefix in checkpoint_include_prefixes
                )
            }
        payload = {
            "phase": phase,
            "step": step,
            "model": state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
            **metadata,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    @torch.no_grad()
    def evaluate():
        if val_loader is None:
            return {}
        cpu_rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        if eval_seed is not None:
            torch.manual_seed(int(eval_seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(eval_seed))
        model.eval()
        try:
            sums, n = {}, 0
            for batch in val_loader:
                if n >= eval_batches:
                    break
                batch = to_device(batch, device)
                with amp_ctx(device, amp):
                    _, metrics = loss_fn(batch)
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                n += 1
            return {key: value / max(n, 1) for key, value in sums.items()}
        finally:
            model.train()
            torch.random.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    model.train()
    iterator = cycle(train_loader)
    last_metrics = {}
    started = time.time()
    try:
        for step in range(start + 1, steps + 1):
            batch = to_device(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            with amp_ctx(device, amp):
                loss, metrics = loss_fn(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{phase} step {step}: non-finite loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            scheduler.step()
            if after_step is not None:
                after_step()
            last_metrics = {key: float(value) for key, value in metrics.items()}
            if step == start + 1 or step % log_every == 0 or step == steps:
                elapsed = max(time.time() - started, 1e-6)
                diagnostics = {
                    f"{phase}/grad_norm": float(grad_norm),
                    f"{phase}/lr": optimizer.param_groups[0]["lr"],
                    f"{phase}/steps_per_s": (step - start) / elapsed,
                    f"{phase}/progress": step / max(steps, 1),
                }
                primary_metrics = metrics_for_logging(last_metrics)
                trainer.log({**primary_metrics, **diagnostics}, step=step)
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "step": step,
                            **primary_metrics,
                            **diagnostics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if val_loader is not None and step % eval_every == 0:
                validation = evaluate()
                validation_log = metrics_for_logging(validation)
                trainer.log(
                    {f"val/{key}": value for key, value in validation_log.items()},
                    step=step,
                )
                monitor_key = training.get("monitor", f"{phase}/total")
                monitor = (
                    validation[monitor_key]
                    if monitor_key in validation
                    else float(loss.detach())
                )
                if monitor < best:
                    best = monitor
                    save(out.parent / "best.pt", step, validation)
            if step % checkpoint_every == 0:
                save(out, step, last_metrics)
        save(out, steps, last_metrics)
        if val_loader is None:
            save(out.parent / "best.pt", steps, last_metrics)
        return out
    finally:
        trainer.finish()
