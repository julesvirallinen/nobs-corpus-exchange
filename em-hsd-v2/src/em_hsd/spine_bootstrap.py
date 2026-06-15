"""Add Johnny t0-1.03/src to sys.path so mechanism/harness/wrapper import cleanly."""

from __future__ import annotations

import sys
from pathlib import Path

_SPINE_SRC = Path(__file__).resolve().parents[2].parent / "Johnny t0-1.03" / "src"
_PATH = str(_SPINE_SRC)
if _PATH not in sys.path:
    sys.path.insert(0, _PATH)


def spine_src() -> Path:
    return _SPINE_SRC
