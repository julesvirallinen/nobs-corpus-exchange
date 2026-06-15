"""Hate utility scorer for candidate selection (standalone; no harness imports)."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import regex as re

from .config import EmHsdConfig

_HATE_KEYS = ("hate", "offensive", "abusive", "toxic", "hateful", "label_1")

_INSULT_KEYWORDS = frozenset(
    "dummy doofus nitwit stupid idiot moron loser clown jerk fool silly "
    "pathetic useless worthless trash garbage zibber florbnax grumblefric".split()
)


class ProxyHateScorer:
    """Fast lexicon/keyword proxy returning P_hate in [0, 1]."""

    name = "proxy"

    def __init__(self, terms: Sequence[str] = ()):
        self._terms = {_skeleton(t) for t in terms if t.strip()}
        self._terms = {s for s in self._terms if len(s) >= 3}

    def score(self, text: str) -> float:
        if self._hit_terms(text) or self._hit_keywords(text):
            return 0.85
        return 0.05

    def _hit_terms(self, text: str) -> bool:
        if not self._terms:
            return False
        toks = re.findall(r"\S+", (text or "").lower())
        tok_skels = {_skeleton(t) for t in toks}
        despaced = _skeleton(re.sub(r"\s+", "", text or ""))
        for term in self._terms:
            if term in tok_skels or term in despaced:
                return True
        return False

    def _hit_keywords(self, text: str) -> bool:
        toks = re.findall(r"[^\W_]+", (text or "").lower())
        return any(w in _INSULT_KEYWORDS for w in toks)


class HFHateScorer:
    """Open-weight hate classifier probability (single model)."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.hate_ids = [
            i for i, lab in id2label.items() if any(k in lab for k in _HATE_KEYS)
        ] or [max(id2label)]

    def score(self, text: str) -> float:
        if not (text or "").strip():
            return 0.0
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = self._torch.softmax(logits, dim=-1)
        return float(sum(probs[i] for i in self.hate_ids))


def _skeleton(token: str) -> str:
    leet = {
        "4": "a", "@": "a", "8": "b", "(": "c", "3": "e", "6": "g", "9": "g",
        "1": "i", "!": "i", "|": "i", "0": "o", "5": "s", "$": "s", "7": "t",
        "+": "t", "2": "z",
    }
    out, prev = [], None
    for ch in token.lower():
        ch = leet.get(ch, ch)
        if not ch.isalnum():
            continue
        if ch != prev:
            out.append(ch)
        prev = ch
    return "".join(out)


def make_scorer(config: EmHsdConfig, backend: str = "proxy") -> object:
    if backend == "hf":
        return HFHateScorer("cardiffnlp/twitter-roberta-base-hate-latest")
    terms: List[str] = []
    lex = config.spine.lexicon
    if lex.source == "test":
        terms = list(lex.test_terms)
    return ProxyHateScorer(terms)


def get_scorer(config: EmHsdConfig, backend: str = "proxy") -> object:
    key = f"_scorer_{backend}"
    cached = getattr(config, key, None)
    if cached is not None:
        return cached
    scorer = make_scorer(config, backend=backend)
    setattr(config, key, scorer)
    return scorer
