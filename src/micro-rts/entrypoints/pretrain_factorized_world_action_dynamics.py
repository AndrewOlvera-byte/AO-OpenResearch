"""Pretrain the factorized world-action dynamics (``causal_world_action_v1``).

Thin shim over the generic ``pretrain`` dispatcher; resolves to the registered
``WorldActionDynamicsTrainer`` via ``model.type: causal_world_action_dynamics``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from entrypoints.pretrain import main  # noqa: E402

DEFAULT_EXP = (
    "micro-rts/paper/incomplete_info/causal_world_action_v1/"
    "pretrain_factorized_world_action_dynamics"
)

if __name__ == "__main__":
    main(default_exp=DEFAULT_EXP)
