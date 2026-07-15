"""Shared plumbing for the DreamerV4 offline pretraining entrypoints.

The tokenizer and dynamics entrypoints share their config-vs-CLI resolution,
model construction, dataset glob resolution, LR schedule, and the latent-scale
measurement that links the two phases (the flow objective runs in unit-RMS
latent space; phase 1 measures the scale, phase 2 consumes it).
"""

from __future__ import annotations

import contextlib
import glob
import math

import torch


def setup_backend(device) -> None:
    """One-time CUDA backend switches for the pretrain loops (no-op on CPU).

    TF32 matmuls/convs: fp32 ops run on tensor cores at ~bf16 throughput with
    fp32 accumulate — free speed for everything autocast doesn't cover (and the
    whole run when ``amp`` is off). cudnn.benchmark: the conv trunks see fixed
    shapes every step, so kernel autotuning pays for itself immediately.
    """
    if getattr(device, "type", str(device)) != "cuda":
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # Prefer fused scaled-dot-product attention kernels. PyTorch still falls
    # back safely when a GPU/mask shape cannot use FlashAttention.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


def amp_ctx(device, enabled: bool):
    """bf16 autocast for the pretrain forward passes; nullcontext when off/CPU.

    bf16 is what makes SDPA dispatch to the FlashAttention kernel (fp32 SDPA
    falls back to the math/mem-efficient paths), and it needs no GradScaler.
    ``cache_enabled=False`` mirrors DreamerRLTrainer._amp_ctx: the loops mix
    no-grad passes (frozen tokenizer encode, val probes) with grad passes over
    the same weights; autocast's weight cache would reuse a no-grad bf16 cast
    inside a later grad-requiring forward and silently detach the graph.
    """
    if enabled and getattr(device, "type", str(device)) == "cuda":
        return torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        )
    return contextlib.nullcontext()


def make_adam(params, lr: float, device, eps: float = 1e-5, weight_decay: float = 0.0):
    """Adam with the fused CUDA kernel when available (one multi-tensor kernel
    per step instead of a Python loop over parameters); plain Adam otherwise."""
    fused = getattr(device, "type", str(device)) == "cuda"
    cls = torch.optim.AdamW if weight_decay else torch.optim.Adam
    try:
        return cls(params, lr=lr, eps=eps, weight_decay=weight_decay, fused=fused)
    except (RuntimeError, TypeError):
        return cls(params, lr=lr, eps=eps, weight_decay=weight_decay)


def resolve_data(pattern: str) -> str:
    """Newest match of a dataset glob (timestamped names sort by time)."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"no dataset matches {pattern!r}")
    return matches[-1]


def build_model(cfg, obs_shape, action_nvec, device):
    from core.registry import build

    model_cfg = {k: v for k, v in cfg.model.items() if k != "type"}
    model = build(
        "model",
        type=cfg.model.get("type", "dreamerv4"),
        obs_shape=obs_shape,
        action_nvec=action_nvec,
        device=str(device),
        **model_cfg,
    )
    return model, model_cfg


def pick(cli, cfg_val, default):
    """CLI flag wins when passed; else the config value; else the built-in default."""
    if cli is not None:
        return cli
    return cfg_val if cfg_val is not None else default


def make_lr_scheduler(
    opt,
    total_steps: int,
    warmup_steps: int = 0,
    min_frac: float = 0.1,
    stages=None,
):
    """Linear warmup to the optimizer's base LR, then cosine decay to
    ``min_frac`` of it by ``total_steps``. Step it once per optimizer step.

    ``stages`` optionally reproduces a continuation schedule inside one
    resumable run. Each item supplies ``end_step``, ``start_frac``, and
    ``end_frac`` relative to the optimizer's base LR. A later stage may also set
    ``warmup_steps`` to ramp smoothly from the preceding floor to its restart LR.
    """
    warmup_steps = max(int(warmup_steps), 0)
    parsed_stages = []
    if stages:
        previous_end = warmup_steps
        for raw in stages:
            stage = dict(raw)
            end = int(stage["end_step"])
            if end <= previous_end:
                raise ValueError("LR stage end_step values must increase")
            start_frac = float(stage["start_frac"])
            end_frac = float(stage["end_frac"])
            restart_warmup = int(stage.get("warmup_steps", 0))
            if start_frac < 0.0 or end_frac < 0.0:
                raise ValueError("LR stage fractions must be non-negative")
            if restart_warmup < 0 or restart_warmup >= end - previous_end:
                raise ValueError("LR stage warmup_steps must fit inside its stage")
            parsed_stages.append(
                (previous_end, end, start_frac, end_frac, restart_warmup)
            )
            previous_end = end
        if previous_end != int(total_steps):
            raise ValueError(
                "the final LR stage end_step must equal total_steps "
                f"({previous_end} != {total_steps})"
            )

    def factor(step):  # LambdaLR calls with 0-based step count
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        if parsed_stages:
            for index, (
                start,
                end,
                start_frac,
                end_frac,
                restart_warmup,
            ) in enumerate(parsed_stages):
                # The exact shared boundary belongs to the preceding stage.
                if step <= end:
                    if index and step == start:
                        return parsed_stages[index - 1][3]
                    if (
                        index
                        and restart_warmup
                        and step <= start + restart_warmup
                    ):
                        previous_frac = parsed_stages[index - 1][3]
                        progress = (step - start) / restart_warmup
                        return previous_frac + (
                            start_frac - previous_frac
                        ) * progress
                    cosine_start = start + (restart_warmup if index else 0)
                    progress = (step - cosine_start) / (end - cosine_start)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return end_frac + (start_frac - end_frac) * cosine
            return parsed_stages[-1][3]
        if total_steps <= warmup_steps:
            return 1.0
        t = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_frac + (1.0 - min_frac) * 0.5 * (
            1.0 + math.cos(math.pi * min(t, 1.0))
        )

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


@torch.no_grad()
def measure_latent_scale(tokenizer, loader, device, batches: int = 32) -> float:
    """RMS of the tokenizer's latents over ``batches`` loader batches — the
    normalization constant the world model's flow objective divides by."""
    total, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        z = tokenizer.encode(batch["obs"].to(device))
        total += float(z.pow(2).sum())
        count += z.numel()
    if count == 0:
        raise RuntimeError("measure_latent_scale: loader yielded no batches")
    return math.sqrt(total / count)
