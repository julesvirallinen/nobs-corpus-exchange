"""Process-level HuggingFace model cache.

All callers that load the same model_id get back the *same* tokenizer+model
objects, so each checkpoint is loaded from disk at most once per process.
Thread-safe via double-checked locking.
"""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()

# model_id -> (AutoTokenizer, AutoModelForSequenceClassification)
_seq_classifiers: dict[str, tuple[Any, Any]] = {}

# model_id -> SentenceTransformer
_sentence_transformers: dict[str, Any] = {}


def get_sequence_classifier(model_id: str) -> tuple[Any, Any]:
    """Return (tokenizer, model) for *model_id*, loading once."""
    if model_id in _seq_classifiers:
        return _seq_classifiers[model_id]
    with _lock:
        if model_id not in _seq_classifiers:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSequenceClassification.from_pretrained(model_id)
            model.eval()
            _seq_classifiers[model_id] = (tok, model)
    return _seq_classifiers[model_id]


def get_sentence_transformer(model_id: str) -> Any:
    """Return a SentenceTransformer for *model_id*, loading once."""
    if model_id in _sentence_transformers:
        return _sentence_transformers[model_id]
    with _lock:
        if model_id not in _sentence_transformers:
            from sentence_transformers import SentenceTransformer
            _sentence_transformers[model_id] = SentenceTransformer(model_id)
    return _sentence_transformers[model_id]
