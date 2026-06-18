"""Layer 2 — Stylometric priors: interface definition.

Layer 2 adjusts the Layer 1 token routes using Biber-style stylometric priors,
boosting protection for tokens that carry author-identifying style. See
``TRIAGE-DP/layer-02-stylometric-priors.md``.

The canonical protocol lives in ``em_hsd.interfaces.triage``; re-exported here.
"""

from __future__ import annotations

from em_hsd.interfaces.triage import StylometricPrior

__all__ = ["StylometricPrior"]
