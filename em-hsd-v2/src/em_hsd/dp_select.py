"""Exponential mechanism selection with optional custom sensitivity."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import spine_bootstrap  # noqa: F401
from mechanism import dp


def select_rewrite(
    candidates: Sequence[str],
    scores: Sequence[float],
    epsilon: float,
    clip: float,
    rng: np.random.Generator,
    sensitivity: Optional[float] = None,
) -> Tuple[str, dp.Selection]:
    if len(candidates) != len(scores):
        raise ValueError("candidates and scores must have equal length")
    if not candidates:
        raise ValueError("no candidates to select from")

    clipped = dp.clip_logits(np.asarray(scores, dtype=np.float64), clip)
    delta = float(sensitivity) if sensitivity is not None else 2.0 * abs(clip)
    if delta <= 0:
        delta = 1.0
    probs = dp.exponential_weights(clipped, epsilon, delta)
    idx = int(rng.choice(len(probs), p=probs))
    sel = dp.Selection(
        index=idx,
        probs=probs,
        clipped=clipped,
        epsilon=float(epsilon),
    )
    return candidates[idx], sel
