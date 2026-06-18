"""Layer 2 + Layer 3 + composed-pipeline wiring tests (offline, no models)."""

from __future__ import annotations

from mechanism.rng import make_row_rng

from em_hsd.interfaces.triage import TokenRoute
from triage_dp.layer2_stylometric.biber_prior import BiberStylometricPrior
from triage_dp.layer3_calibration.calibrate_optimizer import CalibrateTOOptimizer
from triage_dp.pipeline import TriageDpPipeline, build_pipeline


def test_biber_prior_flags_identity_carriers(cfg):
    prior = BiberStylometricPrior()
    routes = prior.boost("you should go downtown honestly", [], cfg)
    by_tok = {r.token: r for r in routes}
    for tok in ("you", "downtown", "honestly"):
        assert tok in by_tok
        assert by_tok[tok].action == "sanitize"
        assert by_tok[tok].biber_boost > 0
        assert by_tok[tok].reason.startswith("biber:")
    assert "should" not in by_tok  # ordinary content untouched


def test_biber_prior_does_not_override_layer1(cfg):
    l1 = [TokenRoute(token="you", start=0, end=3, quadrant="Q1", action="protect",
                     protected_override=True)]
    routes = BiberStylometricPrior().boost("you are here", l1, cfg)
    you = [r for r in routes if r.token == "you"]
    assert len(you) == 1 and you[0].quadrant == "Q1"  # Layer 1 decision kept


def test_calibrate_optimizer_empty_dev_is_noop(cfg):
    out = CalibrateTOOptimizer().optimize([], cfg)
    assert out.epsilon_total is None and out.tau_sem_min is None


def test_calibrate_optimizer_with_dev_returns_overrides(cfg):
    cfg.spine.rng = make_row_rng(0, run_seed="t")
    out = CalibrateTOOptimizer(trials=3, seed=0).optimize(
        [("you are a dummy", "a1"), ("the weather is nice", "a2")], cfg
    )
    assert out.epsilon_total is not None
    assert out.tau_sem_min is not None


def test_build_pipeline_selects_real_layers(cfg):
    cfg.triage_dp.enabled = False
    p = build_pipeline(cfg)
    assert type(p.triage).__name__ == "NoOpTriageRouter"

    cfg.triage_dp.enabled = True
    p = build_pipeline(cfg)
    assert type(p.triage).__name__ == "SaliencyTriageRouter"
    assert type(p.stylometric).__name__ == "BiberStylometricPrior"
    assert type(p.calibration).__name__ == "CalibrateTOOptimizer"


class _StubRouter:
    """Layer 1 stub: protect 'dummy' without loading a model."""

    def route_tokens(self, text, config):
        routes = []
        for m in __import__("re").finditer(r"\S+", text):
            if m.group().lower().strip(".,") == "dummy":
                routes.append(TokenRoute(token=m.group(), start=m.start(), end=m.end(),
                                         quadrant="Q1", action="protect",
                                         protected_override=True))
        return routes


def test_composed_pipeline_l1_protects_l2_sanitizes(cfg):
    """End to end: L1 keeps 'dummy' verbatim; L2 forces 'you' to be rewritten."""
    cfg.spine.rng = make_row_rng(0, run_seed="t")
    pipe = TriageDpPipeline(cfg, triage=_StubRouter(), stylometric=BiberStylometricPrior())
    _, audit = pipe.sanitize("you are such a dummy")
    assert "dummy" in audit["layer1_protected"]
    assert "you" in audit["layer2_boosted"]
    # The protected token survives into the token-sanitised intermediate.
    assert "dummy" in audit["x_priv"].split()
