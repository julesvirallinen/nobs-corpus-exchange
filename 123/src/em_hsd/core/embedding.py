"""Text encoder for semantic floor and near-duplicate pruning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from em_hsd.core.config import EmbeddingSettings, EmHsdConfig


class SimpleEncoder:
    """Deterministic bag-of-char encoder (no downloads)."""

    name = "simple"

    def encode(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            return texts
        vecs = []
        for t in texts:
            counts: dict[str, int] = {}
            if t is None:
                text = ""
            elif isinstance(t, np.ndarray):
                vecs.append(np.asarray(t, dtype=np.float64))
                continue
            else:
                text = str(t)
            for ch in text.lower():
                if ch.isalnum() or ch.isspace():
                    counts[ch] = counts.get(ch, 0) + 1
            if not counts:
                vecs.append(np.zeros(256, dtype=np.float64))
                continue
            idx = np.array([ord(c) % 256 for c in counts], dtype=np.int32)
            w = np.array(list(counts.values()), dtype=np.float64)
            v = np.zeros(256, dtype=np.float64)
            np.add.at(v, idx, w)
            norm = np.linalg.norm(v)
            vecs.append(v / norm if norm > 0 else v)
        return np.vstack(vecs)

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom <= 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


class SentenceTransformerEncoder:
    name = "hf"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str] | np.ndarray) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            return texts
        return np.asarray(
            self._model.encode(list(texts), convert_to_numpy=True),
            dtype=np.float64,
        )

    def cosine(self, a: str, b: str) -> float:
        va, vb = self.encode([a, b])
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        if denom <= 0:
            return 0.0
        return float(np.dot(va, vb) / denom)


def make_encoder(settings: EmbeddingSettings, *, allow_downloads: bool = True) -> object:
    backend = settings.backend
    if backend == "simple":
        return SimpleEncoder()
    if backend == "hf":
        if not allow_downloads:
            raise RuntimeError(
                "Downloads are disabled; cannot load SentenceTransformer model."
            )
        return SentenceTransformerEncoder(settings.model)
    # auto
    if not allow_downloads:
        return SimpleEncoder()
    try:
        return SentenceTransformerEncoder(settings.model)
    except Exception:
        return SimpleEncoder()


def get_encoder(config: EmHsdConfig) -> object:
    from em_hsd.core.resources import ResourceManager
    return ResourceManager(config).encoder()
