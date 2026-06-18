from __future__ import annotations

from em_hsd.interfaces.mock import NoOpStylometricPrior

#: Default stylometric prior used by the pipeline when none is supplied.
DefaultStylometricPrior = NoOpStylometricPrior

__all__ = ["DefaultStylometricPrior"]
