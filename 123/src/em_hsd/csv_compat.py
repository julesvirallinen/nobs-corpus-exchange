"""Backward-compatible shim for legacy `em_hsd.csv_compat` imports."""

from __future__ import annotations

from em_hsd.io.csv_io import (
    CANONICAL_COLUMNS,
    PRESERVED_COLUMNS,
    DiffCheckError,
    append_csv_row_compat,
    assert_preserved_compat,
    read_csv_compat,
    write_canonical_csv,
    write_canonical_privatized_csv,
    write_csv_compat,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "PRESERVED_COLUMNS",
    "DiffCheckError",
    "append_csv_row_compat",
    "assert_preserved_compat",
    "read_csv_compat",
    "write_canonical_csv",
    "write_canonical_privatized_csv",
    "write_csv_compat",
]
