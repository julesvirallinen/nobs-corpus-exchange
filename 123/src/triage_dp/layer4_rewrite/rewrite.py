from __future__ import annotations

from typing import Any

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute
from em_hsd.layer4.orchestrator import Layer4Orchestrator


class EmHsdRewriteLayer:
    def __init__(self, orchestrator: Layer4Orchestrator | None = None) -> None:
        self._orchestrator = orchestrator or Layer4Orchestrator()

    def rewrite(
        self,
        text: str,
        config: EmHsdConfig,
        *,
        token_routes: list[TokenRoute] | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        return self._orchestrator.privatize(
            text,
            config,
            layer1_routes=token_routes,
            layer3_overrides=overrides or None,
        )


#: Default rewrite layer used by the pipeline when none is supplied.
DefaultRewriteLayer = EmHsdRewriteLayer

__all__ = ["EmHsdRewriteLayer", "DefaultRewriteLayer"]
