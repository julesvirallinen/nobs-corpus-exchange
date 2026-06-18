"""TRIAGE-DP pipeline entry point — connects Layers 1–4.

This is the composition root: it wires the four layers together and runs one
input string end to end.

    Layer 1  triage        route_tokens(text)           -> token routes
    Layer 2  stylometric   boost(text, routes)          -> boosted routes
    Layer 3  calibration   optimize(dev_rows)           -> theta overrides
    Layer 4  rewrite        rewrite(text, routes, …)    -> (selected, audit)

Each layer is swappable: pass a custom implementation to the constructor, or
rely on the per-layer defaults (no-op for 1–3, EM-HSD v2 for 4). The default
configuration therefore reproduces standalone Layer 4 behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from em_hsd import load_em_hsd_config
from em_hsd.core.config import EmHsdConfig
from triage_dp.layer1_triage import DefaultTriageRouter, TokenRoute, TriageRouter
from triage_dp.layer2_stylometric import DefaultStylometricPrior, StylometricPrior
from triage_dp.layer3_calibration import DefaultTOOptimizer, TOOptimizer
from triage_dp.layer4_rewrite import DefaultRewriteLayer, RewriteLayer

# Keys an OptimizedConfig may override on Layer 4.
_OVERRIDE_KEYS = (
    "epsilon_total",
    "epsilon_split",
    "hate_floor_delta",
    "tau_sem_min",
    "min_edit_ratio",
    "tau_dup",
    "generation_temperature",
)


class TriageDpPipeline:
    """Compose Layers 1–4 of the TRIAGE-DP pipeline over a single config."""

    def __init__(
        self,
        config: EmHsdConfig,
        *,
        triage: TriageRouter | None = None,
        stylometric: StylometricPrior | None = None,
        calibration: TOOptimizer | None = None,
        rewrite: RewriteLayer | None = None,
    ) -> None:
        self.config = config
        self.triage: TriageRouter = triage or DefaultTriageRouter()
        self.stylometric: StylometricPrior = stylometric or DefaultStylometricPrior()
        self.calibration: TOOptimizer = calibration or DefaultTOOptimizer()
        self.rewrite: RewriteLayer = rewrite or DefaultRewriteLayer()

    @classmethod
    def from_config(cls, config_path: str, **layers: Any) -> TriageDpPipeline:
        """Build a pipeline from a YAML config path."""
        return cls(load_em_hsd_config(config_path), **layers)

    def _calibrate(self, dev_rows: Sequence[tuple[str, str]]) -> dict[str, Any]:
        """Run Layer 3 only in TRIAGE-DP mode; return non-None overrides."""
        if not self.config.is_triage_dp_mode():
            return {}
        opt = self.calibration.optimize(list(dev_rows), self.config)
        return {
            key: value
            for key in _OVERRIDE_KEYS
            if (value := getattr(opt, key, None)) is not None
        }

    def sanitize(
        self,
        text: str,
        *,
        original_text: str | None = None,
        dev_rows: Sequence[tuple[str, str]] = (),
    ) -> tuple[str, dict[str, Any]]:
        """Run one input string through Layers 1→4.

        ``original_text`` is the pre-sanitized raw text used for token routing
        (defaults to ``text``). ``dev_rows`` feed Layer 3 calibration when the
        config enables TRIAGE-DP mode.
        """
        cfg = self.config
        source = original_text if original_text is not None else text

        # Layer 1: route tokens.
        routes: list[TokenRoute] = self.triage.route_tokens(source, cfg)
        # Layer 2: stylometric prior — always runs; it adds its own routes for
        # identity carriers even when Layer 1 found no salient hate tokens.
        routes = self.stylometric.boost(source, routes, cfg)
        # Layer 3: calibrate Layer 4 hyperparameters (TRIAGE-DP mode only).
        overrides = self._calibrate(dev_rows)
        # Layer 4: sentence-level rewrite under the exponential mechanism.
        return self.rewrite.rewrite(
            text,
            cfg,
            token_routes=routes or None,
            overrides=overrides or None,
        )


def build_pipeline(config: EmHsdConfig) -> TriageDpPipeline:
    """Construct the pipeline for *config*.

    When ``triage_dp.enabled`` is set, wires the real Layers 1–3 (cross-saliency
    router, Biber stylometric prior, trade-off calibrator). Otherwise returns the
    NoOp-default pipeline (standalone Layer 4 behaviour).
    """
    if not getattr(config.triage_dp, "enabled", False):
        return TriageDpPipeline(config)

    from triage_dp.layer1_triage.saliency_router import SaliencyTriageRouter
    from triage_dp.layer2_stylometric.biber_prior import BiberStylometricPrior
    from triage_dp.layer3_calibration.calibrate_optimizer import CalibrateTOOptimizer

    return TriageDpPipeline(
        config,
        triage=SaliencyTriageRouter(),
        stylometric=BiberStylometricPrior(),
        calibration=CalibrateTOOptimizer(),
    )


__all__ = ["TriageDpPipeline", "build_pipeline"]
