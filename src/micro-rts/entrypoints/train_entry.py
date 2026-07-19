"""Entrypoint: build a PPO run from an experiment config and train (or smoke-test).

Usage:
    python train_entry.py --exp micro-rts/rl/ppo/base_rlFS_expert
    python train_entry.py --exp micro-rts/rl/ppo/base_rlFS_expert --no-wandb
    python train_entry.py --exp micro-rts/rl/ppo/base_rlFS_expert --test
    python train_entry.py --exp micro-rts/rl/ppo/base_rlFS_expert \
        --set training.ppo.lr=1e-3 --set training.iters=500 --set run.seed=7

Flags:
    --exp        experiment name under configs/exp (required), e.g. micro-rts/rl/ppo/base_rlFS_expert
    --test       single-batch smoke test (no W&B, no checkpoint)
    --no-wandb   disable remote W&B logging (log() becomes a no-op)
    --device     override device (cpu/cuda/auto)
    --set K=V    dotted-path hyperparameter override, repeatable
                 (e.g. --set training.ppo.clip=0.1)

All components (model, curriculum) are built from the config via the registry, so
importing the modules below is what populates the registry (the ``@register``
decorators only fire on import).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Put both the micro-rts package dir and src/ (for `core.*`) on the path so this
# script runs directly, not just under pytest.
_HERE = Path(__file__).resolve()
_PKG = _HERE.parents[1]          # src/micro-rts
_SRC = _HERE.parents[2]          # src
for p in (str(_PKG), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.config import Config  # noqa: E402

# Import for registry side effects (model + curriculum registration).
import models.cnn_mlp_policy  # noqa: E402,F401
import models.masked_policy  # noqa: E402,F401
import models.gridnet_policy  # noqa: E402,F401
import environments.curriculum.PPOCurriculum  # noqa: E402,F401
from trainers.PPOTrainer import PPOTrainer  # noqa: E402
from trainers.BaseTrainer import resolve_device  # noqa: E402


def build_trainer(args) -> PPOTrainer:
    cfg = Config.from_experiment(args.exp)
    cfg.apply_overrides(args.set)
    trainer = PPOTrainer(cfg=cfg)
    if args.device:
        trainer.device = resolve_device(args.device)
    if args.no_wandb:
        trainer.use_wandb = False
    if args.wandb_key:
        trainer._wandb_key = args.wandb_key
    if args.resume or args.resume_from:
        # Resume from a full checkpoint (model + optimizer + step) under the run dir.
        # --resume picks "latest"; --resume-from names a tag (e.g. best, step_65536000).
        trainer._resume_from = args.resume_from or "latest"
    return trainer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MicroRTS PPO trainer")
    parser.add_argument("--exp", required=True, help="experiment name, e.g. micro-rts/rl/ppo/base_rlFS_expert")
    parser.add_argument("--test", action="store_true", help="single-batch smoke test (no W&B / no save)")
    parser.add_argument("--no-wandb", action="store_true", help="disable remote W&B logging")
    parser.add_argument("--wandb-key", default=None, metavar="KEY",
                        help="W&B API key (else uses WANDB_API_KEY env / saved ~/.netrc; "
                             "prompts if none). A provided key is saved in the container.")
    parser.add_argument("--device", default=None, help="override device (cpu/cuda/auto)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from the run's latest.pt (model + optimizer + step)")
    parser.add_argument("--resume-from", default=None, metavar="TAG",
                        help="resume from a specific checkpoint tag, e.g. best or step_65536000")
    parser.add_argument("--set", action="append", default=[], metavar="K=V",
                        help="dotted-path hyperparam override, repeatable (e.g. training.ppo.lr=1e-3)")
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
