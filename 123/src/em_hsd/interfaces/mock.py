from __future__ import annotations

from collections.abc import Sequence

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import (
    OptimizedConfig,
    TokenRoute,
)


class NoOpTriageRouter:
    """Returns empty routes; causes standalone Layer 4 behavior."""

    def route_tokens(self, text: str, config: EmHsdConfig) -> list[TokenRoute]:
        return []


class NoOpStylometricPrior:
    """Returns routes unchanged."""

    def boost(
        self, text: str, token_routes: Sequence[TokenRoute], config: EmHsdConfig
    ) -> list[TokenRoute]:
        return list(token_routes)


class NoOpTOOptimizer:
    """Returns config unchanged."""

    def optimize(
        self, dev_rows: Sequence[tuple[str, str]], config: EmHsdConfig
    ) -> OptimizedConfig:
        return OptimizedConfig()
