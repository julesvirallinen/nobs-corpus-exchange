from __future__ import annotations

from collections.abc import Sequence

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import (
    OptimizedConfig,
    StylometricPrior,
    TokenRoute,
    TOOptimizer,
    TriageRouter,
)


class _MockRouter:
    def route_tokens(self, text: str, config: EmHsdConfig) -> list[TokenRoute]:
        return [TokenRoute(token="x", start=0, end=1, quadrant="Q2", action="sanitize")]


class _MockPrior:
    def boost(
        self, text: str, token_routes: Sequence[TokenRoute], config: EmHsdConfig
    ) -> list[TokenRoute]:
        return list(token_routes)


class _MockOptimizer:
    def optimize(
        self, dev_rows: Sequence[tuple[str, str]], config: EmHsdConfig
    ) -> OptimizedConfig:
        return OptimizedConfig(epsilon_total=9.0)


def test_router_protocol():
    router: TriageRouter = _MockRouter()
    routes = router.route_tokens("hello", None)
    assert len(routes) == 1
    assert routes[0].quadrant == "Q2"


def test_prior_protocol():
    prior: StylometricPrior = _MockPrior()
    routes = [TokenRoute(token="x", start=0, end=1, quadrant="Q2", action="sanitize")]
    assert len(prior.boost("hello", routes, None)) == 1


def test_optimizer_protocol():
    opt: TOOptimizer = _MockOptimizer()
    cfg = opt.optimize([], None)
    assert cfg.epsilon_total == 9.0


def test_optimized_config_apply():
    from em_hsd import load_em_hsd_config

    base = load_em_hsd_config("configs/em-hsd-v2-test.yaml")
    base.em_hsd_v2.epsilon_total = 18.0
    override = OptimizedConfig(epsilon_total=9.0, hate_floor_delta=0.25)
    updated = override.apply(base)
    assert updated.em_hsd_v2.epsilon_total == 9.0
    assert updated.em_hsd_v2.hate_floor_delta == 0.25
