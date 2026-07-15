"""Ensure the Atari package resolves ahead of any sibling env package.

``src/atari`` and ``src/micro-rts`` both expose top-level ``models`` /
``environments`` / ``collectors`` / ``loss`` packages. The root pyproject puts
``src/micro-rts`` on ``pythonpath`` for the default (MicroRTS) test run, so when the
Atari suite is invoked we must insert ``src/atari`` at the *front* of ``sys.path`` so
``import models`` / ``import environments`` resolve to the Atari ones. Run the Atari
suite in its own process (do not co-collect with the MicroRTS tests).
"""

import sys
from pathlib import Path

_ATARI = str(Path(__file__).resolve().parents[1])   # src/atari
_SRC = str(Path(__file__).resolve().parents[2])      # src
for p in (_SRC, _ATARI):                              # insert so _ATARI ends up first
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
