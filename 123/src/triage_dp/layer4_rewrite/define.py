"""Layer 4 — Sentence-level rewrite (EM-HSD v2): interface definition.

Layer 4 is the sentence-level rewriter: given a (possibly token-sanitized)
text plus the upstream Layer 1 routes and Layer 3 overrides, it generates
candidate paraphrases and selects one under the exponential mechanism, falling
back to the safe ``x_priv`` baseline when no candidate is valid. This is the
EM-HSD v2 mechanism. See ``TRIAGE-DP/layer-04--sentence-level-v2.md``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute


@runtime_checkable
class RewriteLayer(Protocol):
    """Contract for the Layer 4 sentence-level rewriter."""

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
