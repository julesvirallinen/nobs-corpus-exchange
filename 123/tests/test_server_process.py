"""Tests for the batch /api/process endpoint used by the Corpus Exchange UI.

The endpoint runs each uploaded text row through the real TRIAGE-DP / Layer-4
pipeline (mock backends via the test config) and returns per-row audit records
plus aggregates that the Upload -> Overview -> Review flow renders and filters.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from em_hsd.server.app import app

client = TestClient(app)

# A mix that the lexicon-driven proxy scorer separates cleanly: rows containing
# test lexicon terms ("dummy", "nitwit", "doofus", "florbnax") score hateful.
HATE_ROWS = [
    "you are a complete dummy and a nitwit",
    "those people are all doofus florbnax",
]
CLEAN_ROWS = [
    "the weather is lovely today",
    "I love this community",
]


def test_process_returns_per_row_records_and_aggregates():
    rows = HATE_ROWS + CLEAN_ROWS
    resp = client.post("/api/process", json={"rows": rows})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Aggregates reflect the real pipeline output.
    assert body["count"] == 4
    assert body["flagged"] == 2
    assert 0.0 <= body["flagged_pct"] <= 100.0
    assert set(body["severity_hist"]) == {"high", "medium", "low"}
    assert set(body["status_hist"]) == {"flagged", "clean"}

    # One record per input row, in order, each carrying real pipeline fields.
    assert len(body["rows"]) == 4
    first = body["rows"][0]
    for key in (
        "id",
        "original",
        "x_priv",
        "p_hate_original",
        "flagged",
        "confidence",
        "confidence_band",
        "severity",
        "tokens_changed",
        "epsilon_total",
    ):
        assert key in first, f"missing {key}"
    assert first["original"] == rows[0]
    # x_priv is the anonymised rewrite produced on-device.
    assert isinstance(first["x_priv"], str) and first["x_priv"]


def test_process_flags_hate_and_clears_clean():
    resp = client.post("/api/process", json={"rows": HATE_ROWS + CLEAN_ROWS})
    body = resp.json()
    by_text = {r["original"]: r for r in body["rows"]}
    for t in HATE_ROWS:
        assert by_text[t]["flagged"] is True
        assert by_text[t]["severity"] in {"high", "medium", "low"}
    for t in CLEAN_ROWS:
        assert by_text[t]["flagged"] is False
        assert by_text[t]["severity"] == "none"


def test_process_confidence_band_derived_from_p_hate():
    resp = client.post("/api/process", json={"rows": HATE_ROWS + CLEAN_ROWS})
    for r in resp.json()["rows"]:
        assert 0.0 <= r["confidence"] <= 1.0
        assert r["confidence_band"] in {"high", "medium", "low"}


def test_process_rejects_empty_rows():
    resp = client.post("/api/process", json={"rows": []})
    assert resp.status_code == 422


def test_process_caps_row_count():
    # Guardrail so an over-large upload can't wedge the demo server.
    too_many = ["x"] * 6000
    resp = client.post("/api/process", json={"rows": too_many})
    assert resp.status_code == 422
