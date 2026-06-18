"""TRIAGE-DP Layer4 adapter wrapping the EM-HSD Layer 4 orchestrator."""

from __future__ import annotations

from typing import Any

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.mock import (
    NoOpStylometricPrior,
    NoOpTOOptimizer,
    NoOpTriageRouter,
)
from em_hsd.interfaces.triage import (
    StylometricPrior,
    TokenRoute,
    TOOptimizer,
    TriageRouter,
)
from em_hsd.layer4.orchestrator import Layer4Orchestrator


class TriageDPLayer4:
    """Temporary adapter that exposes Layer 4 as the TRIAGE-DP sentence-level fallback.

    This lives in ``em_hsd.interfaces`` so the future full TRIAGE-DP repo can
    import it from 123 without needing a separate ``triage_dp`` package yet.
    """

    def __init__(
        self,
        config: EmHsdConfig,
        *,
        router: TriageRouter | None = None,
        prior: StylometricPrior | None = None,
        optimizer: TOOptimizer | None = None,
    ):
        self.config = config
        self.router = router or NoOpTriageRouter()
        self.prior = prior or NoOpStylometricPrior()
        self.optimizer = optimizer or NoOpTOOptimizer()
        self.orchestrator = Layer4Orchestrator()

    def sanitize(
        self,
        text: str,
        *,
        original_text: str | None = None,
        token_routes: list[TokenRoute] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Run the Layer 4 pipeline in TRIAGE-DP composed mode.

        ``original_text`` is the pre-token-sanitized raw text from upstream.
        ``token_routes`` are the Layer 1 (and optionally Layer 2) decisions.
        """
        cfg = self.config
        routes = token_routes
        if routes is None and original_text is not None:
            routes = self.router.route_tokens(original_text, cfg)
        elif routes is None:
            routes = self.router.route_tokens(text, cfg)

        if routes:
            routes = self.prior.boost(original_text or text, routes, cfg)

        overrides: dict[str, Any] = {}
        if cfg.is_triage_dp_mode():
            opt = self.optimizer.optimize([], cfg)
            overrides = {
                k: v for k, v in {
                    "epsilon_total": opt.epsilon_total,
                    "epsilon_split": opt.epsilon_split,
                    "hate_floor_delta": opt.hate_floor_delta,
                    "tau_sem_min": opt.tau_sem_min,
                    "min_edit_ratio": opt.min_edit_ratio,
                    "tau_dup": opt.tau_dup,
                    "generation_temperature": opt.generation_temperature,
                }.items()
                if v is not None
            }

        return self.orchestrator.privatize(
            text,
            cfg,
            layer1_routes=routes,
            layer3_overrides=overrides or None,
        )
