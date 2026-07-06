"""Deprecated alias for ``train_entry.py`` — kept so existing commands still work.

Prefer ``train_entry.py``, which is the canonical PPO entrypoint (supports --exp,
--test, --no-wandb, --device, and repeatable --set K=V overrides).
"""

from __future__ import annotations

from train_entry import main

if __name__ == "__main__":
    main()
