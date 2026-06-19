"""Layer 3 default implementation.

The default optimizer returns an empty ``OptimizedConfig`` (no overrides), so
Layer 4 runs with its configured defaults. Plug a real trade-off optimizer in
here when available.
"""

from __future__ import annotations

from em_hsd.interfaces.mock import NoOpTOOptimizer

#: Default trade-off optimizer used by the pipeline when none is supplied.
DefaultTOOptimizer = NoOpTOOptimizer

__all__ = ["DefaultTOOptimizer"]
