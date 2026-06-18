from __future__ import annotations

import sys
from pathlib import Path

from em_hsd.core.paths import SpinePathResolver

_SPINE_SRC = SpinePathResolver.resolve()
_PATH = str(_SPINE_SRC)
if _PATH not in sys.path:
    sys.path.insert(0, _PATH)


def spine_src() -> Path:
    return _SPINE_SRC
