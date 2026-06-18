"""Layer 3 — trade-off calibration optimizer.

Wraps :class:`em_hsd.calibrate.CalibrateRunner` (random search over ε_total /
hate_floor_delta / tau_sem_min against a local trade-off proxy) as a
:class:`~em_hsd.interfaces.triage.TOOptimizer`, so the composed pipeline tunes
Layer 4 hyperparameters on a small dev set instead of returning the NoOp empty
config.

Honest scope: the trade-off proxy in ``calibrate.py`` is a deterministic
stand-in, not the Johnny harness. With no dev rows there is nothing to
calibrate, so it returns an empty :class:`OptimizedConfig` (Layer 4 defaults).
"""

from __future__ import annotations

from collections.abc import Sequence

from em_hsd.calibrate import CalibrateRunner
from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import OptimizedConfig

_DEFAULT_TRIALS = 20


class CalibrateTOOptimizer:
    """Random-search trade-off calibrator (real, dev-set driven)."""

    def __init__(self, trials: int | None = None, seed: int = 0) -> None:
        self._trials = trials
        self._seed = seed

    def optimize(
        self,
        dev_rows: Sequence[tuple[str, str]],
        config: EmHsdConfig,
    ) -> OptimizedConfig:
        rows = list(dev_rows)
        if not rows:
            return OptimizedConfig()  # no dev set -> keep Layer 4 defaults
        trials = self._trials
        if trials is None:
            layer3 = getattr(config.triage_dp, "layer3", {}) or {}
            trials = int(layer3.get("trials", _DEFAULT_TRIALS))
        summary, _ = CalibrateRunner(
            config, rows, trials=trials, seed=self._seed
        ).run()
        theta = summary.get("best_theta", {}) or {}
        return OptimizedConfig(
            epsilon_total=theta.get("epsilon_total"),
            hate_floor_delta=theta.get("hate_floor_delta"),
            tau_sem_min=theta.get("tau_sem_min"),
            extra={"best_metrics": summary.get("best_metrics", {})},
        )


__all__ = ["CalibrateTOOptimizer"]
