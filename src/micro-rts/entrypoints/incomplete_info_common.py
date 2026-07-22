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
    """Thin shim: delegate to ``PretrainTrainer`` (the loop now lives there).

    Kept so entrypoints not yet migrated to a registered ``PretrainTrainer``
    subclass keep working unchanged.  New stages should subclass PretrainTrainer.
    """
    from trainers.PretrainTrainer import PretrainTrainer

    return PretrainTrainer.from_prebuilt(
        cfg,
        args,
        model,
        loss_fn,
        train_loader,
        val_loader,
        device,
        phase=phase,
        metadata=metadata,
        after_step=after_step,
        trainable_checkpoint_only=trainable_checkpoint_only,
        checkpoint_include_prefixes=checkpoint_include_prefixes,
    )
