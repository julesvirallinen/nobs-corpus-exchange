"""Layer 1 — Cross-saliency triage: interface definition.

Layer 1 routes every token of the input into a quadrant (Q1–Q4) and an action
(keep / sanitize / protect / rewrite), deciding *what* downstream layers must
protect. See ``TRIAGE-DP/layer-01-cross-saliency-triage.md``.

The canonical protocol and data shape live in ``em_hsd.interfaces.triage``;
this module re-exports them so the pipeline package is the single structural
home for the layer contract.
"""

from __future__ import annotations

from em_hsd.interfaces.triage import TokenRoute, TriageRouter

__all__ = ["TokenRoute", "TriageRouter"]
