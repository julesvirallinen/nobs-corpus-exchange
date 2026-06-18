"""JSONL audit writer for EM-HSD CLI runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class AuditJsonlWriter:
    """Append one JSON object per line to an audit file."""

    def __init__(self, path: str | None = None, *, append: bool = False):
        self.path = Path(path) if path else None
        self._append = append
        self._fh = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            self._fh = self.path.open(mode, encoding="utf-8", newline="\n")

    def write(self, obj: dict[str, Any]) -> None:
        """Serialize *obj* as one JSON line."""
        if self._fh is None:
            return
        self._fh.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> AuditJsonlWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
