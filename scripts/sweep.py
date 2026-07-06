"""Throughput sweep — find the optimal collection setup, pushing the env to its limit.

Runs the Runner across configs (overlap on/off, num_envs, horizon) with the real CNN
policy + PPO update, printing agent steps/s for each.

Each config runs in its own subprocess because JPype allows only one JVM per process
(no restart), so a single process can't build more than one env. The driver re-invokes
this script with ``--one`` per config and collects the rows.

    docker exec -i ao-research python /workspace/scripts/sweep.py
"""

from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, "/workspace/src/micro-rts")

GRID = [
    dict(backend="serial", num_envs=64, horizon=128),
    dict(backend="serial", num_envs=128, horizon=128),
    dict(backend="serial", num_envs=256, horizon=128),
    dict(backend="serial", num_envs=256, horizon=256),
]


def run_one(backend, num_envs, horizon, overlap) -> None:
    import torch
    from collectors.runner import RunConfig, Runner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = RunConfig(backend=backend, num_envs=num_envs, horizon=horizon, overlap=overlap,
                    policy="cnn", device=device, iters=12, epochs=4, minibatches=4)
    runner = Runner(cfg)
    try:
        runner.run()           # warm up cudnn/JIT/JVM
        m = runner.run()
    finally:
        runner.close()
    print(f"{backend:<13}{num_envs:>9}{horizon:>8}{str(overlap):>8}"
          f"{m['sps']:>12,.0f}{m['iters_per_s']:>9.2f}")


def driver() -> None:
    import torch
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    print(f"{'backend':<13}{'num_envs':>9}{'horizon':>8}{'overlap':>8}{'steps/s':>12}{'iters/s':>9}")
    print("-" * 59)
    for base in GRID:
        for overlap in (False, True):
            args = [sys.executable, __file__, "--one", base["backend"],
                    str(base["num_envs"]), str(base["horizon"]), str(overlap)]
            r = subprocess.run(args, capture_output=True, text=True)
            row = [ln for ln in r.stdout.splitlines() if ln.startswith(base["backend"])]
            print(row[0] if row else f"{base['backend']} {base['num_envs']} FAILED:\n{r.stderr[-400:]}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        backend, num_envs, horizon, overlap = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "True"
        run_one(backend, num_envs, horizon, overlap)
    else:
        driver()
