from __future__ import annotations

from em_hsd.interfaces.mock import NoOpTOOptimizer

#: Default trade-off optimizer used by the pipeline when none is supplied.
DefaultTOOptimizer = NoOpTOOptimizer

__all__ = ["DefaultTOOptimizer"]
