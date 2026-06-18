from __future__ import annotations

import re
from typing import Any

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute

_DEFAULT_THRESHOLD = 0.15
_DEFAULT_MODEL = "unitary/unbiased-toxic-roberta"
_WORD = re.compile(r"\S+")


class SaliencyTriageRouter:
    def __init__(self, scorer: Any | None = None) -> None:
        # ``scorer`` exposes hate_prob(text)->float. None = lazy-load the
        # occlusion classifier on first use.
        self._scorer = scorer
        self._load_failed = False

    def _get_scorer(self, config: EmHsdConfig) -> Any | None:
        if self._scorer is not None:
            return self._scorer
        if self._load_failed:
            return None
        saliency = getattr(config.spine, "saliency", None)
        model = getattr(saliency, "model", None) or _DEFAULT_MODEL
        try:
            from mechanism.saliency import OcclusionSaliency

            self._scorer = OcclusionSaliency(model)
        except Exception:  # missing HF extra / model / offline-uncached
            self._load_failed = True
            return None
        return self._scorer

    def _threshold(self, config: EmHsdConfig) -> float:
        layer1 = getattr(config.triage_dp, "layer1", {}) or {}
        val = layer1.get("threshold")
        if val is None:
            saliency = getattr(config.spine, "saliency", None)
            val = getattr(saliency, "threshold", None)
        return float(val) if val is not None else _DEFAULT_THRESHOLD

    def route_tokens(self, text: str, config: EmHsdConfig) -> list[TokenRoute]:
        scorer = self._get_scorer(config)
        if scorer is None or not (text or "").strip():
            return []
        threshold = self._threshold(config)
        base = scorer.hate_prob(text)
        routes: list[TokenRoute] = []
        for m in _WORD.finditer(text):
            start, end = m.start(), m.end()
            occluded = text[:start] + (" " * (end - start)) + text[end:]
            drop = base - scorer.hate_prob(occluded)
            if drop >= threshold:
                routes.append(
                    TokenRoute(
                        token=m.group(),
                        start=start,
                        end=end,
                        quadrant="Q1",
                        action="protect",
                        protected_override=True,
                        reason=f"classifier-salient (Δhate={drop:.3f})",
                    )
                )
        return routes


__all__ = ["SaliencyTriageRouter"]
