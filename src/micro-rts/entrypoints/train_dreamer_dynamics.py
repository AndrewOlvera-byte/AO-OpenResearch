"""Train world-model dynamics selected by ``trainer.type``."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from trainers.DynamicsTrainers import (  # noqa: E402,F401
    load_pretrained_action_tokenizer,
    main,
)


if __name__ == "__main__":
    main()
