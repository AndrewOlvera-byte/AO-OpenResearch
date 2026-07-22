"""Train an opponent tokenizer, intent prior, or belief latent trainer."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
for root in (HERE.parents[1], HERE.parents[2]):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from entrypoints.pretrain import main  # noqa: E402


if __name__ == "__main__":
    main()
