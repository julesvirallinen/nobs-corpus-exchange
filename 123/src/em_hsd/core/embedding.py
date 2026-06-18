"""Text encoder for semantic floor and near-duplicate pruning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from em_hsd.core.config import EmbeddingSettings, EmHsdConfig


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
    if backend != "hf":
        raise ValueError(
            f"unknown embedding.backend {backend!r} (expected hf)"
        )
    if not allow_downloads:
        raise RuntimeError(
            "Downloads are disabled; cannot load SentenceTransformer model."
        )
    return SentenceTransformerEncoder(settings.model)


def get_encoder(config: EmHsdConfig) -> object:
    from em_hsd.core.resources import ResourceManager
    return ResourceManager(config).encoder()
