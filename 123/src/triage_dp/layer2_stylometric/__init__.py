"""Layer 2 — Stylometric priors (Biber-style route boosts)."""

from __future__ import annotations

from triage_dp.layer2_stylometric.default import DefaultStylometricPrior
from triage_dp.layer2_stylometric.define import StylometricPrior

__all__ = ["DefaultStylometricPrior", "StylometricPrior"]
