"""Hate utility scorer for candidate selection (standalone; no harness imports)."""

from __future__ import annotations

from collections.abc import Sequence

import regex as re

from em_hsd.core.config import EmHsdConfig

_HATE_KEYS = ("hate", "offensive", "abusive", "toxic", "hateful", "label_1")

_INSULT_KEYWORDS = frozenset(
    ["dummy", "doofus", "nitwit", "stupid", "idiot", "moron", "loser", "clown", "jerk", "fool", "silly", "pathetic", "useless", "worthless", "trash", "garbage", "zibber", "florbnax", "grumblefric"]
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
        return any(term in tok_skels or term in despaced for term in self._terms)

    def _hit_keywords(self, text: str) -> bool:
        toks = re.findall(r"[^\W_]+", (text or "").lower())
        return any(w in _INSULT_KEYWORDS for w in toks)


def _label_index(id2label: dict, score_label: str) -> int:
    target = score_label.lower()
    for idx, lab in id2label.items():
        if str(lab).lower() == target:
            return int(idx)
    for idx, lab in id2label.items():
        if target in str(lab).lower():
            return int(idx)
    raise ValueError(f"score_label {score_label!r} not found in id2label {id2label!r}")


def _score_from_logits(logits, model_config, score_label: str, torch) -> float:
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


class HFToxicityScorer:
    """Open-weight toxicity classifier probability in [0, 1]."""

    def __init__(self, model_name: str, score_label: str = "toxicity"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self._score_label = score_label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score(self, text: str) -> float:
        if not (text or "").strip():
            return 0.0
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        return _score_from_logits(
            logits, self.model.config, self._score_label, self._torch,
        )


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


def make_scorer(config: EmHsdConfig, *, allow_downloads: bool = True) -> object:
    util = config.utility
    if util.backend == "hf":
        if not allow_downloads:
            raise RuntimeError(
                "Downloads are disabled; cannot load HuggingFace toxicity model."
            )
        return HFToxicityScorer(util.model, score_label=util.score_label)
    terms: list[str] = []
    lex = config.spine.lexicon
    if lex.source == "test":
        terms = list(lex.test_terms)
    return ProxyHateScorer(terms)


def get_scorer(config: EmHsdConfig) -> object:
    from em_hsd.core.resources import ResourceManager
    return ResourceManager(config).scorer()
