"""Layer 1 default implementation.

The default router is a no-op that emits no token routes, which makes the
pipeline fall back to standalone Layer 4 behaviour. Plug a real cross-saliency
triage router in here (or pass one to the pipeline entry point) when available.
"""

from __future__ import annotations

from em_hsd.interfaces.mock import NoOpTriageRouter

#: Default router used by the pipeline when none is supplied.
DefaultTriageRouter = NoOpTriageRouter

__all__ = ["DefaultTriageRouter"]
