"""Per-row randomness policy.

The privacy story depends on getting this exactly right (see SEED_POLICY.md):

* There is NO global seed. A single fixed global seed would make the whole
  mechanism reproducible from one number and undermine the privacy story.
* Every row gets its OWN independent ``numpy`` Generator.
    - Production: fresh OS entropy per row (irreproducible by design).
    - Debug: per-row seed = SHA-256(RUN_SEED, row_index). Reproducible AND
      mutually independent across rows. Debug mode is for development only and
      MUST NOT be used to produce a submission.
* Independence holds regardless of batching/order: a row's generator depends
  only on (RUN_SEED, row_index), never on its neighbours.
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np


def _derive_seed(run_seed: str, row_index: int) -> int:
    """Deterministically derive a 128-bit seed from (run_seed, row_index)."""
    payload = f"{run_seed}\x00{row_index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:16], "big")


def production_rng() -> np.random.Generator:
    """Fresh, OS-entropy-seeded generator. Independent on every call."""
    return np.random.default_rng()  # seeds from os entropy when given no seed


def debug_rng(run_seed: str, row_index: int) -> np.random.Generator:
    """Reproducible per-row generator keyed by (run_seed, row_index)."""
    return np.random.default_rng(_derive_seed(run_seed, row_index))


def make_row_rng(row_index: int, run_seed: Optional[str] = None) -> np.random.Generator:
    """Return the generator for ``row_index``. ``run_seed=None`` -> production
    (fresh entropy); a string -> debug (reproducible, still per-row independent)."""
    if run_seed is None:
        return production_rng()
    return debug_rng(run_seed, row_index)
