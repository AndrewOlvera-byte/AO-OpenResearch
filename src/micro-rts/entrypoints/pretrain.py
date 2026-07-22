"""Generic pretraining entrypoint.

Dispatches to a registered ``PretrainTrainer`` subclass by the config's
``model.type`` (or ``training.trainer``), builds it from config, and runs
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

# Ordered inference for configs that key the stage by which model sub-block is
# present rather than an explicit ``model.type`` (most specific first).
_SUBBLOCK_TRAINER = [
    ("factorized_dynamics", "causal_world_action_dynamics"),
    ("predictive_belief", "causal_world_action_encoder"),
    ("belief_dynamics", "belief_dynamics"),
    ("flow", "joint_flow_dynamics"),
    ("intent_prior", "opponent_intent_prior"),
    ("opponent_tokenizer", "incomplete_opponent_plan_tokenizer"),
    ("self_action_tokenizer", "incomplete_self_action_tokenizer"),
    ("ego_tokenizer", "incomplete_ego_tokenizer"),
]


def resolve_trainer_type(cfg):
    """Explicit ``training.trainer`` / ``model.type`` win; else infer from the
    distinctive model sub-block."""
    model = cfg.model or {}
    explicit = (cfg.training or {}).get("trainer") or model.get("type")
    if explicit:
        return explicit
    for key, trainer_type in _SUBBLOCK_TRAINER:
        if key in model:
            return trainer_type
    raise ValueError(
        "cannot resolve trainer: set training.trainer or model.type in the config"
    )


def main(argv=None, default_exp=""):
    parser = common_parser(__doc__, default_exp)
    args = parser.parse_args(argv)
    cfg = load_config(args)
    trainer = build("trainer", type=resolve_trainer_type(cfg), cfg=cfg, args=args)
    return trainer.smoke_test() if args.smoke else trainer.train()


if __name__ == "__main__":
    main()
