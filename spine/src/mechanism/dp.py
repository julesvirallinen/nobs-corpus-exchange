from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


def clip_logits(logits: np.ndarray, clip: float) -> np.ndarray:    return np.clip(np.asarray(logits, dtype=np.float64), -abs(clip), abs(clip))


def exponential_weights(clipped_scores: np.ndarray, epsilon: float,
                        sensitivity: float) -> np.ndarray:
    """Exponential-mechanism probabilities over ALREADY-CLIPPED scores.

    Numerically stable. ``sensitivity`` is Δ (the score range, = 2·clip).
    """
    scores = np.asarray(clipped_scores, dtype=np.float64)
    if scores.size == 0:
        return scores
    if sensitivity <= 0:
        sensitivity = 1.0
    scale = float(epsilon) / (2.0 * sensitivity)
    z = scores * scale
    z = z - np.max(z)
    w = np.exp(z)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.full(scores.shape, 1.0 / scores.size)
    return w / total


@dataclass
class Selection:
    index: int
    probs: np.ndarray
    clipped: np.ndarray
    epsilon: float


def select_index(raw_scores: Sequence[float], epsilon: float, clip: float,
                 rng: np.random.Generator) -> Selection:
    """Clip ``raw_scores``, build exponential-mechanism probabilities at ``epsilon``
    and draw one index with ``rng``. ``sensitivity`` is fixed at ``2·clip``."""
    clipped = clip_logits(np.asarray(raw_scores, dtype=np.float64), clip)
    sensitivity = 2.0 * abs(clip)
    probs = exponential_weights(clipped, epsilon, sensitivity)
    idx = int(rng.choice(len(probs), p=probs))
    return Selection(index=idx, probs=probs, clipped=clipped, epsilon=float(epsilon))


def select(candidates: Sequence[str], raw_scores: Sequence[float], epsilon: float,
           clip: float, rng: np.random.Generator) -> Tuple[str, Selection]:    sel = select_index(raw_scores, epsilon, clip, rng)
    return candidates[sel.index], sel
