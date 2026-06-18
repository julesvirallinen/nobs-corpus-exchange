"""Layer 4 default implementation: EM-HSD v2 rewrite.

Thin adapter exposing the existing :class:`em_hsd.layer4.orchestrator.Layer4Orchestrator`
through the :class:`triage_dp.layer4_rewrite.define.RewriteLayer` contract. The
orchestrator caches its scorer/encoder/proposer, so reusing one instance keeps
models warm across calls.
"""

from __future__ import annotations

from typing import Any

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute
from em_hsd.layer4.orchestrator import Layer4Orchestrator


class EmHsdRewriteLayer:
    """Layer 4 rewriter backed by the EM-HSD v2 orchestrator."""

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
