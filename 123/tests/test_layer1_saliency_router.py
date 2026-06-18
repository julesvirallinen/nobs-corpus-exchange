"""Tests for the real Layer 1 cross-saliency triage router."""

from __future__ import annotations

import pytest

from triage_dp.layer1_triage.saliency_router import SaliencyTriageRouter


class _StubScorer:
    """Hate score that only the token 'muslims' drives — so occluding it drops
    the score and every other occlusion leaves it unchanged."""

    def hate_prob(self, text: str) -> float:
        return 0.9 if "muslims" in text.lower() else 0.1


def test_router_protects_the_salient_token(cfg):
    router = SaliencyTriageRouter(scorer=_StubScorer())
    routes = router.route_tokens("muslims are terrorists", cfg)
    assert len(routes) == 1
    r = routes[0]
    assert r.token == "muslims"
    assert r.quadrant == "Q1"
    assert r.action == "protect"
    assert r.protected_override is True
    assert r.start == 0 and r.end == len("muslims")


def test_router_emits_nothing_for_clean_text(cfg):
    router = SaliencyTriageRouter(scorer=_StubScorer())
    assert router.route_tokens("the weather is lovely today", cfg) == []


def test_router_degrades_to_no_routes_when_model_unavailable(cfg):
    # No scorer injected and the loader fails -> empty routes (standalone L4).
    router = SaliencyTriageRouter()
    router._load_failed = True  # simulate load failure without touching HF
    assert router.route_tokens("muslims are terrorists", cfg) == []


def test_router_respects_threshold(cfg):
    # Drop for 'muslims' is 0.8; a threshold above that yields no protect route.
    cfg.triage_dp.layer1 = {"threshold": 0.95}
    router = SaliencyTriageRouter(scorer=_StubScorer())
    assert router.route_tokens("muslims are terrorists", cfg) == []


@pytest.mark.skipif(
    True,
    reason="real occlusion model integration covered by composed-pipeline test",
)
def test_router_with_real_model(cfg):  # pragma: no cover
    router = SaliencyTriageRouter()
    routes = router.route_tokens("muslims are terrorists and should be banned", cfg)
    assert any(r.token.lower().startswith("muslim") for r in routes)
