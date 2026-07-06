"""Throughput benchmark: env steps/second for the in-process env vs the multiprocess pool.

Run inside the container:
    docker exec -i ao-research python /workspace/scripts/bench_sps.py --num-envs 24 --workers 4

Reports agent steps/s = num_envs * iters / wall_time, driven by RandomPolicy (so the
measurement reflects env+transport cost, not model cost).
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "/workspace/src/micro-rts")

import torch  # noqa: E402

from collectors.vecpool import MultiprocessPool  # noqa: E402
from environments.microrts_env import EnvConfig, MicroRTSVecEnv  # noqa: E402
from models.random_policy import RandomPolicy  # noqa: E402


def bench(env, iters: int, warmup: int = 10) -> float:
    policy = RandomPolicy(env.action_nvec)
    trans = env.reset()
    for i in range(warmup + iters):
        if i == warmup:
            t0 = time.perf_counter()
        action = policy.step(trans["obs"])["action"]
        trans = env.step(action)
    dt = time.perf_counter() - t0
    return env.num_envs * iters / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-envs", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    cfg = EnvConfig(num_envs=args.num_envs, max_steps=2000, mode="bot",
                    bots=("randomBiasedAI",))

    serial = MicroRTSVecEnv(cfg)
    sps_serial = bench(serial, args.iters)
    serial.close()
    print(f"SerialPool    (1 JVM, {args.num_envs} envs): {sps_serial:,.0f} steps/s")

    pool = MultiprocessPool(cfg, num_workers=args.workers)
    sps_pool = bench(pool, args.iters)
    pool.close()
    print(f"MultiprocessPool ({args.workers} JVMs, {args.num_envs} envs): {sps_pool:,.0f} steps/s")
    print(f"speedup: {sps_pool / sps_serial:.2f}x")


if __name__ == "__main__":
    main()
