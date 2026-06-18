"""Layer 3 — Trade-off (TO) calibration: interface definition.

Layer 3 calibrates the Layer 4 hyperparameters (epsilon split, hate floor,
semantic thresholds, …) offline on a small dev set, emitting an
``OptimizedConfig`` of theta overrides. See
``TRIAGE-DP/layer-03-to-calibration.md``.

The canonical protocol and output shape live in ``em_hsd.interfaces.triage``;
re-exported here.
"""

from __future__ import annotations

from em_hsd.interfaces.triage import OptimizedConfig, TOOptimizer

__all__ = ["OptimizedConfig", "TOOptimizer"]
