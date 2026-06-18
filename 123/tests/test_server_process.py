"""Tests for the batch /api/process endpoint used by the Corpus Exchange UI.

The endpoint runs each uploaded text row through the real TRIAGE-DP / Layer-4
pipeline and returns per-row audit records plus aggregates that the Upload ->
Overview -> Review flow renders and filters.

Structural tests pass config='em-hsd-v2-test.yaml' explicitly so they pass in
offline CI without model downloads. Semantic hate-detection tests are skipped
when unitary/unbiased-toxic-roberta is not cached or EM_HSD_ALLOW_DOWNLOADS
is not set.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from em_hsd.core.policy import DownloadPolicy
from em_hsd.server.app import app
from em_hsd.server.classifier import classifier

client = TestClient(app)

# Phrases that RoBERTa reliably flags as toxic.
HATE_ROWS = [
    "Muslims are terrorists and should be banned from the country",
    "those gay people are disgusting and unnatural",
]
CLEAN_ROWS = [
    "the weather is lovely today",
    "I love this community",
]

needs_roberta = pytest.mark.skipif(
    not classifier.available or not DownloadPolicy.is_allowed(),
    reason="unitary/unbiased-toxic-roberta not available or EM_HSD_ALLOW_DOWNLOADS not set",
)


def test_process_returns_per_row_records_and_aggregates():
    # Use the no-download test config — this test checks response shape, not scorer quality.
    rows = ["you are a complete dummy", "the weather is nice"]
    resp = client.post("/api/process", json={"rows": rows, "config": "em-hsd-v2-test.yaml"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["count"] == 2
    assert 0.0 <= body["flagged_pct"] <= 100.0
    assert set(body["severity_hist"]) == {"high", "medium", "low"}
    assert set(body["status_hist"]) == {"flagged", "clean"}

    assert len(body["rows"]) == 2
    first = body["rows"][0]
    for key in (
        "id", "original", "x_priv", "p_hate_original", "flagged",
        "confidence", "confidence_band", "severity", "tokens_changed", "epsilon_total",
    ):
        assert key in first, f"missing {key}"
    assert first["original"] == rows[0]
    assert isinstance(first["x_priv"], str) and first["x_priv"]


@needs_roberta
def test_process_flags_hate_and_clears_clean():
    resp = client.post("/api/process", json={"rows": HATE_ROWS + CLEAN_ROWS})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    by_text = {r["original"]: r for r in body["rows"]}
    for t in HATE_ROWS:
        assert by_text[t]["flagged"] is True
        assert by_text[t]["severity"] in {"high", "medium", "low"}
    for t in CLEAN_ROWS:
        assert by_text[t]["flagged"] is False
        assert by_text[t]["severity"] == "none"


def test_process_confidence_band_derived_from_p_hate():
    rows = ["you are a complete dummy", "the weather is nice"]
    resp = client.post("/api/process", json={"rows": rows, "config": "em-hsd-v2-test.yaml"})
    for r in resp.json()["rows"]:
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["confidence_band"] in {"high", "medium", "low"}


def test_process_rejects_empty_rows():
    resp = client.post("/api/process", json={"rows": []})
    assert resp.status_code == 422


def test_process_caps_row_count():
    too_many = ["x"] * 6000
    resp = client.post("/api/process", json={"rows": too_many})
    assert resp.status_code == 422
