"""Test 5: salient matching — leet/spacing/elongation variants are caught and
canonicalised; a protected token never keeps an idiosyncratic surface form."""

import pytest

from mechanism import privatize
from mechanism.rng import make_row_rng
from mechanism.spine import get_resources

# variants -> canonical (all registered in configs/test.yaml)
VARIANTS = [
    ("zibber", "zibber"),
    ("z1bb3r", "zibber"),
    ("Z1BB3R", "zibber"),
    ("z i b b e r", "zibber"),
    ("ziiibber", "zibber"),
    ("z.i.b.b.e.r", "zibber"),
    ("d00fus", "doofus"),
    ("d u m m y", "dummy"),
    ("n1tw1t", "nitwit"),
    ("grumblefric", "grumblefric"),
    ("g r u m b l e f r i c", "grumblefric"),
]


@pytest.mark.parametrize("variant,canonical", VARIANTS)
def test_lexicon_catches_and_canonicalises(variant, canonical, cfg):
    lexicon = get_resources(cfg)[0]
    spans = lexicon.find_protected_spans(variant)
    assert spans, f"no protected span found for {variant!r}"
    assert any(canon == canonical for _, _, canon in spans), (
        f"{variant!r} did not map to {canonical!r}: {spans}"
    )


@pytest.mark.parametrize("variant,canonical", VARIANTS)
def test_protected_token_loses_idiosyncratic_spelling(variant, canonical, cfg):
    cfg.rng = make_row_rng(1, run_seed="SAL")
    text = f"you are a total {variant} ok"
    _, log = privatize(text, cfg)
    protected = [e for e in log if e["token_class"] == "protected"]
    assert protected, f"no protected token in log for {variant!r}"
    entry = protected[0]
    assert entry["replacement"] == canonical
    assert entry["action"] == "protected+canonicalised"
    # the idiosyncratic surface must NOT survive in the output replacement
    if variant != canonical:
        assert entry["replacement"] != entry["original"]


def test_clean_token_not_protected(cfg):
    cfg.rng = make_row_rng(2, run_seed="SAL")
    _, log = privatize("the weather is lovely today", cfg)
    assert all(e["token_class"] != "protected" for e in log)
