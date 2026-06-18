from __future__ import annotations

from typing import List, Sequence

import numpy as np
import regex as re

_LEET_TO_LETTER = {
    "4": "a", "@": "a", "8": "b", "(": "c", "3": "e", "6": "g", "9": "g",
    "1": "i", "!": "i", "|": "i", "0": "o", "5": "s", "$": "s", "7": "t",
    "+": "t", "2": "z",
}

_INSULT_KEYWORDS = frozenset(
    "dummy doofus nitwit stupid idiot moron loser clown jerk fool silly "
    "pathetic useless worthless trash garbage".split()
)

_HATE_KEYS = ("hate", "offensive", "abusive", "toxic", "hateful", "label_1")
_DEFAULT_HF_MODEL = "unitary/unbiased-toxic-roberta"
_DEFAULT_SCORE_LABEL = "toxicity"


def _skeleton(token: str) -> str:
    out, prev = [], None
    for ch in token.lower():
        ch = _LEET_TO_LETTER.get(ch, ch)
        if not ch.isalnum():
            continue
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


class LexiconProxyClassifier:
    name = "proxy-lexicon"

    def __init__(self, terms: Sequence[str]):
        self._skel_terms = sorted({_skeleton(t) for t in terms if t.strip()})
        self._skel_terms = [s for s in self._skel_terms if len(s) >= 3]

    def _hit(self, text: str) -> bool:
        if not self._skel_terms:
            return False
        toks = re.findall(r"\S+", (text or "").lower())
        tok_skels = {_skeleton(t) for t in toks}
        despaced = _skeleton(re.sub(r"\s+", "", text or ""))
        for term in self._skel_terms:
            if term in tok_skels or term in despaced:
                return True
        return False

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([1 if self._hit(t) else 0 for t in texts], dtype=int)


class KeywordProxyClassifier:
    name = "proxy-keyword"

    def __init__(self, keywords=_INSULT_KEYWORDS):
        self._kw = set(keywords)

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            toks = re.findall(r"[^\W_]+", (t or "").lower())
            out.append(1 if any(w in self._kw for w in toks) else 0)
        return np.asarray(out, dtype=int)


def _label_index(id2label: dict, score_label: str) -> int:
    target = score_label.lower()
    for idx, lab in id2label.items():
        if str(lab).lower() == target:
            return int(idx)
    for idx, lab in id2label.items():
        if target in str(lab).lower():
            return int(idx)
    raise ValueError(f"score_label {score_label!r} not found in id2label {id2label!r}")


def _hate_probability(logits, model_config, score_label: str, torch) -> float:
    problem = getattr(model_config, "problem_type", None) or ""
    id2label = {int(k): v for k, v in model_config.id2label.items()}

    if problem == "multi_label_classification":
        probs = torch.sigmoid(logits)
        idx = _label_index(id2label, score_label)
        return float(probs[idx])

    probs = torch.softmax(logits, dim=-1)
    hate_ids = [
        i for i, lab in id2label.items()
        if any(k in str(lab).lower() for k in _HATE_KEYS)
    ] or [max(id2label)]
    return float(sum(probs[i] for i in hate_ids))


class HFHateClassifier:
    def __init__(self, model_name: str, score_label: str = _DEFAULT_SCORE_LABEL):
        import torch  # noqa: F401
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self._torch = torch
        self.name = model_name
        self._score_label = score_label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        out = []
        for t in texts:
            if not (t or "").strip():
                out.append(0)
                continue
            enc = self.tokenizer(t, return_tensors="pt", truncation=True,
                                 max_length=256)
            with torch.no_grad():
                logits = self.model(**enc).logits[0]
            hate = _hate_probability(logits, self.model.config, self._score_label, torch)
            out.append(1 if hate >= 0.5 else 0)
        return np.asarray(out, dtype=int)


def build_classifiers(backend: str, terms: Sequence[str] = (),
                      hf_models: List[str] = None) -> List[object]:
    if backend == "proxy":
        return [LexiconProxyClassifier(terms), KeywordProxyClassifier()]
    if backend == "hf":
        models = hf_models or [_DEFAULT_HF_MODEL]
        clfs = []
        for m in models:
            try:
                clfs.append(HFHateClassifier(m))
            except Exception as exc:  # pragma: no cover - network/availability
                print(f"WARN: could not load classifier {m!r}: {exc}")
        if not clfs:
            raise RuntimeError("no HF classifiers could be loaded")
        return clfs
    raise ValueError(f"unknown utility backend {backend!r} (expected hf|proxy)")
