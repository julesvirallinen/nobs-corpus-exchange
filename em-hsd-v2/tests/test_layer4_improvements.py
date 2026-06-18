"""Tests for Layer 4 doc-backed improvements (embedding ε₁, prompts, protected merge)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2
from em_hsd.prompts import build_paraphrase_prompt, format_protected_list, resolve_prompt_profile
from em_hsd.resources import merge_protected_terms, protected_from_token_log
from mechanism.mlm import EmbeddingMLM
from mechanism.rng import make_row_rng

ROOT = Path(__file__).resolve().parents[1]


class _StubEncoder:
    """Deterministic unit-norm vectors keyed by word text."""

    _DIM = 8

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        rows = [self._vec(t) for t in texts]
        return np.asarray(rows, dtype=np.float64)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self._DIM, dtype=np.float64)
        for i, ch in enumerate(str(text).lower()):
            v[i % self._DIM] += (ord(ch) % 97) + 1
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v


@pytest.fixture
def stub_embedding_mlm(tmp_path):
    vocab = tmp_path / "candidates.txt"
    vocab.write_text("hello\nworld\nplanet\ngood\nnice\n", encoding="utf-8")
    mlm = EmbeddingMLM(
        embedding_model="stub",
        candidate_vocab=str(vocab),
        hybrid_hf=False,
    )
    mlm._encoder = _StubEncoder()
    vecs = mlm._encoder.encode(mlm.candidate_vocab)
    vecs = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    mlm._vocab_matrix = vecs / norms
    return mlm


def test_embedding_mlm_includes_original(stub_embedding_mlm):
    mlm = stub_embedding_mlm
    cands, scores = mlm.score("", "", "hello", top_k=3, include_original=True)
    assert "hello" in cands
    assert len(scores) == len(cands)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_embedding_mlm_high_utility_neighbor_ranked(stub_embedding_mlm):
    mlm = stub_embedding_mlm
    cands, scores = mlm.score("", "", "hello", top_k=5, include_original=True)
    idx = {w: s for w, s in zip(cands, scores)}
    assert idx["hello"] >= max(s for w, s in idx.items() if w != "hello")


def test_protected_from_token_log_extracts_surfaces():
    log = [
        {"action": "sanitized", "original": "foo", "replacement": "bar"},
        {"action": "protected+canonicalised", "original": "retarded", "replacement": "retarded"},
    ]
    assert protected_from_token_log(log) == ["retarded"]


def test_merge_protected_terms_unions_lexicon_and_saliency():
    canon, skels = merge_protected_terms(["dummy"], ["retarded"])
    assert set(canon) == {"dummy", "retarded"}
    assert len(skels) == 2


def test_format_protected_list_empty_hint():
    hint = format_protected_list([])
    assert "do NOT remove" in hint
    assert "slurs" in hint


def test_resolve_prompt_profile_auto_threshold(cfg):
    cfg.generation.prompt_profile = "auto"
    cfg.generation.hate_p_threshold = 0.35
    assert resolve_prompt_profile(cfg, 0.5) == "hate"
    assert resolve_prompt_profile(cfg, 0.1) == "neutral"


def test_build_paraphrase_prompt_hate_keeps_protected(cfg):
    prompt = build_paraphrase_prompt(
        cfg, ["dummy", "retarded"], "you are a dummy", profile="hate", variant_idx=0,
    )
    assert "dummy" in prompt
    assert "retarded" in prompt
    assert "Privacy (p)" in prompt
    assert "Utility (u)" in prompt


def test_saliency_terms_merge_into_protected_terms(cfg, monkeypatch):
    from mechanism.canonicalize import canonicalize_protected
    from mechanism.tokenize import PROTECTED, WORD

    def _fake_apply(segments, config):
        for seg in segments:
            if seg.kind == WORD and seg.text == "retarded":
                seg.kind = PROTECTED
                seg.canonical = seg.text
                canonicalize_protected(seg)

    monkeypatch.setattr("mechanism.saliency.apply_saliency", _fake_apply)
    cfg.spine.saliency.enabled = True
    cfg.spine.rng = make_row_rng(99, run_seed="test")

    _, audit = privatize_em_hsd_v2("you are retarded and dumb", cfg)
    assert "retarded" in audit["protected_terms_saliency"]
    assert "retarded" in audit["protected_terms"]


def test_production_config_embedding_backend():
    prod = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2.yaml"))
    assert prod.spine.mlm.backend == "embedding"
    assert prod.spine.mlm.top_k == 32
    assert prod.spine.mlm.clip == 1.0
    assert prod.generation.prompt_profile == "auto"
    assert prod.generation.prompt_jitter is True
    assert prod.embedding.backend == "hf"
    assert prod.em_hsd_v2.k_generate == 6
    assert prod.em_hsd_v2.k_max_after_prune == 4


def test_reddit_config_k_and_prompt_settings():
    reddit = load_em_hsd_config(str(ROOT / "configs" / "em-hsd-v2-qwen-reddit.yaml"))
    assert reddit.spine.mlm.backend == "embedding"
    assert reddit.em_hsd_v2.k_generate == 6
    assert reddit.em_hsd_v2.k_max_after_prune == 4
    assert reddit.generation.prompt_jitter is True
