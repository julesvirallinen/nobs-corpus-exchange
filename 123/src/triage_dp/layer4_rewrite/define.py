from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute


@runtime_checkable
class RewriteLayer(Protocol):
    def rewrite(
        self,
        text: str,
        config: EmHsdConfig,
        *,
        token_routes: list[TokenRoute] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Return ``(selected_text, audit)`` for one input string."""
        ...


__all__ = ["RewriteLayer", "TokenRoute"]
