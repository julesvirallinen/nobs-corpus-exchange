"""Layer 1–3 protocol definitions for TRIAGE-DP integration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from em_hsd.core.config import EmHsdConfig


@dataclass
class TokenRoute:
    """Per-token decision produced by Layer 1 (and optionally Layer 2)."""

    token: str
    start: int
    end: int
    quadrant: str  # Q1, Q2, Q3, Q4 from cross-saliency triage
    action: str  # keep | sanitize | protect | rewrite
    epsilon: float | None = None
    reason: str = ""
    protected_override: bool = False
    biber_boost: float = 0.0


@dataclass
class OptimizedConfig:
    """Layer 3 calibration output: theta overrides for Layer 4."""

    epsilon_total: float | None = None
    epsilon_split: float | None = None
    hate_floor_delta: float | None = None
    tau_sem_min: float | None = None
    min_edit_ratio: float | None = None
    tau_dup: float | None = None
    generation_temperature: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def apply(self, config: EmHsdConfig) -> EmHsdConfig:
        """Return a shallow copy of *config* with any non-None overrides applied."""
        cfg = config
        em = cfg.em_hsd_v2
        if self.epsilon_total is not None:
            em.epsilon_total = self.epsilon_total
        if self.epsilon_split is not None:
            em.epsilon_split = self.epsilon_split
        if self.hate_floor_delta is not None:
            em.hate_floor_delta = self.hate_floor_delta
        if self.tau_sem_min is not None:
            em.tau_sem_min = self.tau_sem_min
        if self.min_edit_ratio is not None:
            em.min_edit_ratio = self.min_edit_ratio
        if self.tau_dup is not None:
            em.tau_dup = self.tau_dup
        if self.generation_temperature is not None:
            em.generation_temperature = self.generation_temperature
        return cfg


@runtime_checkable
class TriageRouter(Protocol):
    """Layer 1: cross-saliency triage assigns a quadrant/action per token."""

    def route_tokens(self, text: str, config: EmHsdConfig) -> list[TokenRoute]:
        """Return one TokenRoute per token/span."""
        ...


@runtime_checkable
class StylometricPrior(Protocol):
    """Layer 2: Biber-style prior boosts for token routes."""

    def boost(
        self, text: str, token_routes: Sequence[TokenRoute], config: EmHsdConfig
    ) -> list[TokenRoute]:
        """Return a possibly-modified list of token routes with `biber_boost` set."""
        ...


@runtime_checkable
class TOOptimizer(Protocol):
    """Layer 3: calibrate Layer 4 hyperparameters on a small dev set."""

    def optimize(
        self, dev_rows: Sequence[tuple[str, str]], config: EmHsdConfig
    ) -> OptimizedConfig:
        """*dev_rows* are (original_text, author_id) pairs.

        Return an OptimizedConfig with theta overrides for Layer 4.
        """
        ...
