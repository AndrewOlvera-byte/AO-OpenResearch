from __future__ import annotations

import glob
from pathlib import Path

import torch


def resolve_latest(pattern: str) -> Path:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        p = Path(pattern)
        if p.exists():
            return p
        raise FileNotFoundError(f"no data matches {pattern!r}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def device_from(value: str):
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


@torch.no_grad()
def measure_token_stats(tokenizer, loader, device, batches=32):
    total = total2 = count = None
    for i, batch in enumerate(loader):
        if i >= batches:
            break
        z = tokenizer.encode(
            batch["state"].to(device), batch["globals"].to(device)
        ).float()
        dims = tuple(range(z.ndim - 1))
        s, s2 = z.sum(dims), z.square().sum(dims)
        n = z.numel() // z.shape[-1]
        total = s if total is None else total + s
        total2 = s2 if total2 is None else total2 + s2
        count = n if count is None else count + n
    mean = total / count
    var = (total2 / count - mean.square()).clamp_min(1e-8)
    return mean, var.sqrt()
