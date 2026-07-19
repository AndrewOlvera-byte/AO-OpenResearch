"""Evaluate a trained MicroRTS policy against the scripted bots.

Plays full games (win-loss reward) so the reported numbers are real match
outcomes, not fragments of the dense shaping reward. For each bot we track
wins / losses / draws, win rate, average game length, and an implied Elo
(anchored to a fixed bot rating), plus an aggregate Elo across bots.

Usage:
    # default: eval configs/exp/<exp>'s best checkpoint under its ckpt dir
    python eval_micro_rts.py --exp micro-rts/rl/ppo/base_rlFS_expert

    # override the checkpoint (path may start at checkpoints/, be an absolute
    # path, or a bare tag like "final" / "step_650444800")
    python eval_micro_rts.py --exp micro-rts/rl/ppo/base_rlFS_expert \
        --checkpoint checkpoints/base_rlFS_expert/step_650444800.pt

    python eval_micro_rts.py --exp micro-rts/rl/ppo/base_rlFS_expert \
        --bots coacAI workerRushAI lightRushAI --games 50 --sample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Path setup (run directly, not just under pytest).
_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # src/micro-rts
_SRC = _HERE.parents[2]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch  # noqa: E402

from core.config import Config  # noqa: E402
from core.registry import build  # noqa: E402
import models.cnn_mlp_policy  # noqa: E402,F401  (registry side effect)
import models.masked_policy  # noqa: E402,F401  (registry side effect)
import models.gridnet_policy  # noqa: E402,F401  (registry side effect)
from environments.microrts_env import EnvConfig, MicroRTSVecEnv  # noqa: E402
from evaluation.matchplay import build_report, run_match_eval  # noqa: E402
from rewards.rewards import reward_weight  # noqa: E402
from trainers.BaseTrainer import resolve_device  # noqa: E402

# Win-loss-only reward: terminal reward is +1 win / -1 loss / 0 draw (timeout).
WIN_LOSS_WEIGHT = reward_weight("win_loss")


# --- checkpoint resolution ----------------------------------------------
def resolve_checkpoint(cfg: Config, arg: str | None) -> Path:
    run_dir = Path(cfg.run.get("ckpt_dir", "checkpoints")) / cfg.run.get("name", "run")
    if not arg:
        return run_dir / "best.pt"

    candidates = [
        Path(arg),                              # as given (abs or cwd-relative)
        Path("checkpoints") / arg,              # path starting at checkpoints/
        run_dir / arg,                          # relative to this run's ckpt dir
        run_dir / f"{arg}.pt",                  # bare tag: best / final / step_x
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"checkpoint not found for --checkpoint {arg!r}; tried {candidates}")


def load_policy(cfg: Config, ckpt_path: Path, env, device) -> torch.nn.Module:
    model_cfg = {k: v for k, v in cfg.model.items() if k != "type"}
    policy = build(
        "model", type=cfg.model["type"],
        obs_shape=env.obs_shape, action_nvec=env.action_nvec,
        device=str(device), **model_cfg,
    )
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(payload["model"])
    policy.eval()
    return policy, payload.get("step")


# --- match play ----------------------------------------------------------
def play_matches(policy, bots, games, max_steps, device, envs_per_bot=8,
                 deterministic=True, seed=0, gridnet=False):
    """Build a bot env, play the matches (shared logic), and close the env."""
    torch.manual_seed(seed)
    n = envs_per_bot * len(bots)
    env = MicroRTSVecEnv(EnvConfig(
        num_envs=n, mode="bot", bots=tuple(bots), max_steps=max_steps,
        reward_weight=WIN_LOSS_WEIGHT, gridnet=gridnet,
    ))
    env_bots = [bots[i % len(bots)] for i in range(n)]
    try:
        return run_match_eval(policy, env, env_bots, games, max_steps, device, deterministic)
    finally:
        env.close()


# --- reporting -----------------------------------------------------------
def print_report(report, ckpt_path, step):
    print(f"\n=== MicroRTS eval | {ckpt_path}" + (f" (step {step:,})" if step else "") + " ===")
    print("  metric = full-game outcomes vs scripted bots (win=+1 / loss=-1 / draw=timeout)\n")
    print(f"  {'bot':<16}{'games':>6}{'W':>5}{'L':>5}{'D':>5}{'win%':>8}{'avg_len':>9}{'elo':>8}")
    print("  " + "-" * 62)
    for bot, r in report["per_bot"].items():
        print(f"  {bot:<16}{r['games']:>6}{r['wins']:>5}{r['losses']:>5}{r['draws']:>5}"
              f"{r['win_rate'] * 100:>7.1f}%{r['avg_len']:>9.0f}{r['elo']:>8.0f}")
    o = report["overall"]
    print("  " + "-" * 62)
    print(f"  {'OVERALL':<16}{o['games']:>6}{'':>15}{o['win_rate'] * 100:>7.1f}%"
          f"{'':>9}{o['elo']:>8.0f}\n")


# --- entry ---------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a MicroRTS policy vs scripted bots")
    parser.add_argument("--exp", required=True, help="experiment name, e.g. micro-rts/rl/ppo/base_rlFS_expert")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path (may start at checkpoints/, be absolute, or a bare tag). Default: best.pt")
    parser.add_argument("--bots", nargs="+", default=None, help="opponent bots (default: curriculum eval_bots)")
    parser.add_argument("--games", type=int, default=30, help="games per bot")
    parser.add_argument("--envs-per-bot", type=int, default=8, help="parallel envs per bot")
    parser.add_argument("--max-steps", type=int, default=2000, help="game step limit (timeout -> draw)")
    parser.add_argument("--sample", action="store_true", help="sample actions instead of greedy argmax")
    parser.add_argument("--device", default=None, help="cpu/cuda/auto")
    parser.add_argument("--out", default=None, help="output JSON path (default: <run>/eval_report_<ckpt>.json)")
    args = parser.parse_args(argv)

    cfg = Config.from_experiment(args.exp)
    device = resolve_device(args.device or cfg.run.get("device", "auto"))
    ckpt_path = resolve_checkpoint(cfg, args.checkpoint)

    curriculum = cfg.training["curriculum"]
    bots = args.bots or list(curriculum.get("eval_bots", ["coacAI"]))
    gridnet = bool(curriculum.get("gridnet", False))
    run_dir = Path(cfg.run.get("ckpt_dir", "checkpoints")) / cfg.run.get("name", "run")

    # Build a throwaway env just to size the policy, then load weights.
    sizing_env = MicroRTSVecEnv(EnvConfig(num_envs=len(bots), mode="bot", bots=tuple(bots),
                                          reward_weight=WIN_LOSS_WEIGHT, gridnet=gridnet))
    obs_shape, action_nvec = sizing_env.obs_shape, sizing_env.action_nvec
    policy, step = load_policy(cfg, ckpt_path, sizing_env, device)

    print(f"[eval] exp={args.exp} ckpt={ckpt_path} device={device}\n"
          f"[eval] bots={bots} games/bot={args.games} "
          f"mode={'sample' if args.sample else 'greedy'}")
    t0 = time.perf_counter()
    stats = play_matches(policy, bots, args.games, args.max_steps, device,
                         envs_per_bot=args.envs_per_bot, deterministic=not args.sample,
                         gridnet=gridnet)
    report = build_report(stats)
    report["meta"] = {
        "exp": args.exp, "checkpoint": str(ckpt_path), "step": step,
        "games_per_bot": args.games, "greedy": not args.sample,
        "eval_seconds": time.perf_counter() - t0,
    }
    print_report(report, ckpt_path, step)

    out = Path(args.out) if args.out else run_dir / f"eval_report_{ckpt_path.stem}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
