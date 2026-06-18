from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from em_hsd import load_em_hsd_config
from em_hsd.core.config import EmHsdConfig
from triage_dp.layer1_triage import DefaultTriageRouter, TokenRoute, TriageRouter
from triage_dp.layer2_stylometric import DefaultStylometricPrior, StylometricPrior
from triage_dp.layer3_calibration import DefaultTOOptimizer, TOOptimizer
from triage_dp.layer4_rewrite import DefaultRewriteLayer, RewriteLayer

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
        return cls(load_em_hsd_config(config_path), **layers)

    def _calibrate(self, dev_rows: Sequence[tuple[str, str]]) -> dict[str, Any]:
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
        cfg = self.config
        source = original_text if original_text is not None else text

        routes: list[TokenRoute] = self.triage.route_tokens(source, cfg)
        routes = self.stylometric.boost(source, routes, cfg)
        overrides = self._calibrate(dev_rows)
        return self.rewrite.rewrite(
            text,
            cfg,
            token_routes=routes or None,
            overrides=overrides or None,
        )


def build_pipeline(config: EmHsdConfig) -> TriageDpPipeline:
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
