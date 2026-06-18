from __future__ import annotations

import threading
from typing import Any

MODEL_ID = "unitary/unbiased-toxic-roberta"

TOXICITY_THRESHOLD = 0.5
IDENTITY_THRESHOLD = 0.5

_IDENTITY_TO_CATEGORY = {
    "christian": "religion",
    "jewish": "religion",
    "muslim": "religion",
    "male": "gender",
    "female": "gender",
    "homosexual_gay_or_lesbian": "orientation",
    "black": "ethnicity",
    "white": "ethnicity",
    "psychiatric_or_mental_illness": "other",
}
_HIGH_TYPES = ("threat", "severe_toxicity", "identity_attack")
_MEDIUM_TYPES = ("insult", "obscene", "sexual_explicit")


class ToxicRobertaClassifier:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: str | None = None
        self._torch = None
        self._tok = None
        self._model = None
        self._id2label: dict[int, str] = {}

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return True
        if self._load_error is not None:
            return False
        with self._lock:
            if self._loaded:
                return True
            try:
                import torch
                from em_hsd.core.model_cache import get_sequence_classifier

                self._torch = torch
                self._tok, self._model = get_sequence_classifier(self.model_id)
                self._id2label = {
                    int(k): str(v) for k, v in self._model.config.id2label.items()
                }
                self._loaded = True
                return True
            except Exception as exc:
                self._load_error = str(exc)
                return False

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def _probs(self, text: str) -> dict[str, float]:
        enc = self._tok(
            text or "", return_tensors="pt", truncation=True, max_length=256
        )
        with self._torch.no_grad():
            logits = self._model(**enc).logits[0]
        p = self._torch.sigmoid(logits)
        return {self._id2label[i]: float(p[i]) for i in range(len(p))}

    def classify(self, text: str) -> dict[str, Any]:
        if not self._ensure_loaded():
            raise RuntimeError(f"classifier unavailable: {self._load_error}")
        if not (text or "").strip():
            return {
                "flagged": False,
                "p_hate": 0.0,
                "severity": "none",
                "category": None,
                "confidence": 1.0,
                "scores": {},
            }
        probs = self._probs(text)
        tox = probs.get("toxicity", 0.0)
        flagged = tox >= TOXICITY_THRESHOLD

        ident = {lab: probs.get(lab, 0.0) for lab in _IDENTITY_TO_CATEGORY}
        top_lab = max(ident, key=ident.get)
        category = (
            _IDENTITY_TO_CATEGORY[top_lab]
            if flagged and ident[top_lab] >= IDENTITY_THRESHOLD
            else None
        )

        if not flagged:
            severity = "none"
        elif any(probs.get(t, 0.0) >= 0.5 for t in _HIGH_TYPES):
            severity = "high"
        elif any(probs.get(t, 0.0) >= 0.5 for t in _MEDIUM_TYPES):
            severity = "medium"
        else:
            severity = "low"

        confidence = round(tox if flagged else 1.0 - tox, 3)
        return {
            "flagged": flagged,
            "p_hate": round(tox, 4),
            "severity": severity,
            "category": category,
            "category_label": top_lab if category else None,
            "confidence": confidence,
            "scores": {k: round(v, 4) for k, v in probs.items()},
        }


classifier = ToxicRobertaClassifier()
