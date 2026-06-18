"""Exponential mechanism selection with optional custom sensitivity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mechanism import dp

from em_hsd import spine_bootstrap  # noqa: F401


def select_rewrite(
    candidates: Sequence[str],
    scores: Sequence[float],
    epsilon: float,
    clip: float,
    rng: np.random.Generator,
    sensitivity: float | None = None,
) -> tuple[str, dp.Selection]:
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
