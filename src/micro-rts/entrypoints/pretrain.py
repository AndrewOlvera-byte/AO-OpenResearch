"""Generic pretraining entrypoint.

Dispatches to a registered ``PretrainTrainer`` subclass by ``trainer.type``,
builds it from config, and runs
``smoke_test`` / ``train``.  Every staged pretraining run goes through here:

    python entrypoints/pretrain.py --exp <experiment> [--smoke] [--no-wandb]

Individual named entrypoints (e.g. ``pretrain_factorized_world_action_dynamics``)
are thin shims around this for discoverability.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import registry_imports  # noqa: F401,E402  (fires model/loss/trainer decorators)
from core.registry import build  # noqa: E402
from entrypoints.incomplete_info_common import (  # noqa: E402
    common_parser,
    load_config,
)

def resolve_trainer_type(cfg):
    """Return the explicit trainer type; architecture inference is forbidden."""
    trainer_type = (cfg.trainer or {}).get("type")
    if not trainer_type:
        raise ValueError("cannot resolve trainer: set trainer.type in the config")
    return trainer_type


def main(argv=None, default_exp=""):
    parser = common_parser(__doc__, default_exp)
    args = parser.parse_args(argv)
    cfg = load_config(args)
    trainer = build("trainer", type=resolve_trainer_type(cfg), cfg=cfg, args=args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
