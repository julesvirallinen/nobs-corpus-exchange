"""TRIAGE-DP — differential-privacy text sanitisation pipeline.

One package, four layers, each in its own folder with a ``define`` module
(the layer's interface contract) plus a default implementation:

    layer1_triage/        Layer 1 — cross-saliency token triage
    layer2_stylometric/   Layer 2 — Biber-style stylometric priors
    layer3_calibration/   Layer 3 — trade-off (theta) calibration
    layer4_rewrite/       Layer 4 — sentence-level rewrite (EM-HSD v2)

:mod:`triage_dp.pipeline` is the entry point that connects the layers.

Importing this package pulls in ``em_hsd`` (and transitively ``mechanism``),
so the SPINE source dir must be importable first — see the package README.
"""

from __future__ import annotations

from triage_dp.pipeline import TriageDpPipeline

__all__ = ["TriageDpPipeline"]
