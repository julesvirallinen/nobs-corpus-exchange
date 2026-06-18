from __future__ import annotations

import os
import sys
from pathlib import Path

_SIBLING_NAME = "spine"
_SIBREL_SRC = "src"
_ENV_VAR = "EM_HSD_SPINE_PATH"


def repo_root() -> Path:    # this file: <root>/src/em_hsd/server/_bootstrap.py
    return Path(__file__).resolve().parents[3]


def configs_dir() -> Path:
    return repo_root() / "configs"


def ensure_spine_on_path() -> str | None:
    """Resolve the SPINE src dir and insert it into ``sys.path``.

    Resolution order mirrors :class:`em_hsd.core.paths.SpinePathResolver`:
      1. ``EM_HSD_SPINE_PATH`` env var.
      2. Sibling ``../spine/src`` next to the repo root.

    Returns the resolved path as a string, or ``None`` if not found (in which
    case the spine is assumed to be importable via ``PYTHONPATH`` already).
    """
    candidates: list[Path] = []
    env_path = os.environ.get(_ENV_VAR, "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(repo_root().parent / _SIBLING_NAME / _SIBREL_SRC)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            path_str = str(resolved)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
            return path_str
    return None


# Run on import so ``from em_hsd.server._bootstrap import ...`` is safe only if
# this file is imported standalone first. The launcher imports it by path.
ensure_spine_on_path()
