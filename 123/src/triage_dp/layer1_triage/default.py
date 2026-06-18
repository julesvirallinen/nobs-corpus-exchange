from __future__ import annotations

from em_hsd.interfaces.mock import NoOpTriageRouter

#: Default router used by the pipeline when none is supplied.
DefaultTriageRouter = NoOpTriageRouter

__all__ = ["DefaultTriageRouter"]
