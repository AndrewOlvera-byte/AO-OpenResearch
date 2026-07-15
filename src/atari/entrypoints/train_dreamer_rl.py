"""Entrypoint: build an Atari DreamerV4 run from a config and train (or smoke-test).

Usage:
    python src/atari/entrypoints/train_dreamer_rl.py --exp atari/base_dreamerv4
    python src/atari/entrypoints/train_dreamer_rl.py --exp atari/base_dreamerv4 --test
    python .../train_dreamer_rl.py --exp atari/base_dreamerv4 --no-wandb \
        --set training.env.game=breakout --set training.dreamer.batch_seqs=32

Puts ``src/atari`` and ``src`` on the path so ``import environments.* / models.* /
core.* / shared.*`` resolve. NB: run Atari in its own process — ``src/atari`` and
``src/micro-rts`` both expose top-level ``models``/``environments`` packages, so they
must not share a Python process.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # src/atari
_SRC = _HERE.parents[2]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.config import Config  # noqa: E402
from entrypoints.AtariDreamTrainer import AtariDreamTrainer, resolve_device  # noqa: E402


def build_trainer(args) -> AtariDreamTrainer:
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    trainer = AtariDreamTrainer(cfg)
    if args.device:
        trainer.device = resolve_device(args.device)
    if args.no_wandb:
        trainer.use_wandb = False
    return trainer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Atari DreamerV4 trainer")
    parser.add_argument("--exp", required=True, help="experiment name, e.g. atari/base_dreamerv4")
    parser.add_argument("--test", action="store_true", help="single-iteration smoke test")
    parser.add_argument("--no-wandb", action="store_true", help="disable W&B logging")
    parser.add_argument("--device", default=None, help="override device (cpu/cuda/auto)")
    parser.add_argument("--set", action="append", default=[], metavar="K=V",
                        help="dotted-path override, repeatable")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    trainer = build_trainer(args)
    if args.test:
        print("smoke ok:", trainer.smoke_test())
    else:
        trainer.train()


if __name__ == "__main__":
    main()
