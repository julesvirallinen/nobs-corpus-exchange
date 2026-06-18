"""Layer 2 default implementation.

The default prior returns token routes unchanged. Plug a real Biber-style
stylometric prior in here when available.
"""

from __future__ import annotations

from em_hsd.interfaces.mock import NoOpStylometricPrior

#: Default stylometric prior used by the pipeline when none is supplied.
DefaultStylometricPrior = NoOpStylometricPrior

__all__ = ["DefaultStylometricPrior"]
