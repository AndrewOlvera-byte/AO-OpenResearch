"""Entrypoint: build a DreamerV4 model-based RL run from a config and train.

``training.dreamer.mode`` (or ``--mode``) picks where the actor/critic gradient
comes from — the three runs whose win_rate-vs-env-steps curves measure what
imagination buys (see the ``rl_dreamerv4_*`` configs, all warm-started from the
phase-2 pretrain checkpoint via ``training.dreamer.init_from``):

- ``imagination`` — actor/critic train purely inside the FROZEN pretrained world
  model (the full Dreamer 4 pretrain-then-RL recipe),
- ``hybrid``      — the world model keeps finetuning on fresh replay while the
  actor dreams (classic online Dreamer; collector splices terminal frames),
- ``online``      — actor/critic train on real replay sequences, no world model
  in the loop (the model-free sample-efficiency baseline).

Usage:
    python train_dreamer_rl.py --exp micro-rts/rl/dreamerv4/rl_dreamerv4_imagination
    python train_dreamer_rl.py --exp micro-rts/rl/dreamerv4/rl_dreamerv4_hybrid --test
    python train_dreamer_rl.py --exp micro-rts/rl/dreamerv4/rl_dreamerv4_online --no-wandb \
        --set training.dreamer.horizon=32 --set training.dreamer.amp=false

Flags mirror ``train_entry.py`` (the PPO entrypoint): ``--exp`` selects an
experiment under ``configs/exp``; ``--test`` runs the single-iteration smoke test;
``--no-wandb`` disables logging; ``--device`` overrides the device; repeatable
``--set K=V`` applies dotted-path overrides.

The DreamerV4 model type is registered by importing ``models.dreamer``; the trainer
lives in ``trainers/DreamerRLTrainer.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # src/micro-rts
_SRC = _HERE.parents[2]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.config import Config  # noqa: E402

import models.dreamer  # noqa: E402,F401  (registry side effect)
import models.dreamer_v2  # noqa: E402,F401  (structured agent registry)
from trainers.DreamerRLTrainer import DreamerRLTrainer  # noqa: E402
from trainers.StructuredDreamerRLTrainer import StructuredDreamerRLTrainer  # noqa: E402
from trainers.BaseTrainer import resolve_device  # noqa: E402


def build_trainer(args) -> DreamerRLTrainer | StructuredDreamerRLTrainer:
    cfg = Config.from_experiment(args.exp)
    overrides = list(args.set)
    if args.mode:
        overrides.append(f"training.dreamer.mode={args.mode}")
    cfg.apply_overrides(overrides)
    trainer_cls = (
        StructuredDreamerRLTrainer
        if cfg.model.get("type") == "structured_dreamer"
        else DreamerRLTrainer
    )
    trainer = trainer_cls(cfg)
    if args.device:
        trainer.device = resolve_device(args.device)
    if args.no_wandb:
        trainer.use_wandb = False
    if args.wandb_key:
        trainer._wandb_key = args.wandb_key
    return trainer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MicroRTS DreamerV4 trainer")
    parser.add_argument("--exp", required=True, help="experiment name, e.g. micro-rts/rl/dreamerv4/base_dreamerv4")
    parser.add_argument("--mode", choices=["imagination", "hybrid", "online"], default=None,
                        help="override training.dreamer.mode: actor/critic gradient source "
                             "(imagination = frozen pretrained WM, hybrid = WM keeps training, "
                             "online = real sequences, no WM — the baseline)")
    parser.add_argument("--test", action="store_true", help="single-iteration smoke test (no W&B / no save)")
    parser.add_argument("--no-wandb", action="store_true", help="disable remote W&B logging")
    parser.add_argument("--wandb-key", default=None, metavar="KEY", help="W&B API key")
    parser.add_argument("--device", default=None, help="override device (cpu/cuda/auto)")
    parser.add_argument("--set", action="append", default=[], metavar="K=V",
                        help="dotted-path override, repeatable (e.g. training.dreamer.horizon=64)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    trainer = build_trainer(args)
    if args.test:
        trainer.use_wandb = False
        print("smoke ok:", trainer.smoke_test())
    else:
        trainer.train()


if __name__ == "__main__":
    main()
