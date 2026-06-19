"""Near-duplicate candidate pruning (PrivRewrite §4.1.3 pattern)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from em_hsd.core.config import EmHsdConfig
from em_hsd.core.embedding import get_encoder


class CandidatePruner(Protocol):
    def prune(self, candidates: Sequence[str], config: EmHsdConfig) -> list[str]:
        ...


def prune_candidates(candidates: Sequence[str], config: EmHsdConfig) -> list[str]:
    if not candidates:
        return []
    em = config.em_hsd_v2
    encoder = get_encoder(config)
    vecs = encoder.encode(list(candidates))  # type: ignore[attr-defined]
    if not isinstance(vecs, np.ndarray):
        vecs = np.asarray(vecs)
    kept: list[int] = []
    for i in range(len(candidates)):
        ok = True
        for j in kept:
            denom = np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
            cos = float(np.dot(vecs[i], vecs[j]) / denom) if denom > 0 else 0.0
            if cos >= em.tau_dup:
                ok = False
                break
        if ok:
            kept.append(i)
        if len(kept) >= em.k_max_after_prune:
            break
    return [candidates[i] for i in kept]
