"""Tests for the real multi-label classifier path (classifier='hf').

Skipped automatically when unitary/unbiased-toxic-roberta is not cached /
downloadable, so offline CI stays green; the proxy path is covered separately.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from em_hsd.server.app import app
from em_hsd.server.classifier import classifier

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    not classifier.available,
    reason="unitary/unbiased-toxic-roberta not available (offline / uncached)",
)


def test_hf_assigns_target_group_category_and_severity():
    rows = [
        "Muslims are terrorists and should be banned from the country",
        "those gay people are disgusting and unnatural",
        "I love the new park, great for the whole family",
    ]
    resp = client.post("/api/process", json={"rows": rows, "classifier": "hf"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["classifier"] == "hf"

    by_text = {r["original"]: r for r in body["rows"]}
    religion = by_text[rows[0]]
    assert religion["flagged"] is True
    assert religion["category"] == "religion"
    assert religion["severity"] in {"high", "medium", "low"}

    orientation = by_text[rows[1]]
    assert orientation["flagged"] is True
    assert orientation["category"] == "orientation"

    clean = by_text[rows[2]]
    assert clean["flagged"] is False
    assert clean["category"] is None
    assert clean["severity"] == "none"

    # Aggregates expose the category histogram for the overview.
    assert "category_hist" in body
    assert body["category_hist"].get("religion", 0) >= 1


def test_classifier_classify_shape():
    out = classifier.classify("Muslims are terrorists")
    assert set(out) >= {"flagged", "p_hate", "severity", "category", "confidence"}
    assert out["category"] == "religion"
    assert 0.0 <= out["p_hate"] <= 1.0
