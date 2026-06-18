from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def _derive_seed(run_seed: str, row_index: int) -> int:    payload = f"{run_seed}\x00{row_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], "big")


def production_rng() -> np.random.Generator:    return np.random.default_rng()  # seeds from os entropy when given no seed


def debug_rng(run_seed: str, row_index: int) -> np.random.Generator:    return np.random.default_rng(_derive_seed(run_seed, row_index))


def make_row_rng(row_index: int, run_seed: Optional[str] = None) -> np.random.Generator:
    """Return the generator for ``row_index``. ``run_seed=None`` -> production
    (fresh entropy); a string -> debug (reproducible, still per-row independent)."""
    if run_seed is None:
        return production_rng()
    return debug_rng(run_seed, row_index)
