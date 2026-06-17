"""Hate utility scorer for candidate selection (standalone; no harness imports)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence

import regex as re

from .config import EmHsdConfig

_HATE_KEYS = ("hate", "offensive", "abusive", "toxic", "hateful", "label_1")

# Detoxify unbiased-toxic-roberta heads (Jigsaw unintended bias).
_DETOXIFY_LABELS = (
    "toxicity",
    "severe_toxicity",
    "obscene",
    "threat",
    "insult",
    "identity_attack",
    "sexual_explicit",
)

_INSULT_KEYWORDS = frozenset(
    "dummy doofus nitwit stupid idiot moron loser clown jerk fool silly "
    "pathetic useless worthless trash garbage zibber florbnax grumblefric".split()
)


@dataclass
class HsdAnalysis:
    """HSD model output: primary utility, per-label probs, severity, active labels."""

    p_hate: float
    labels: Dict[str, float] = field(default_factory=dict)
    hs_labels: List[str] = field(default_factory=list)
    severity: str = "none"
    severity_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _empty_analysis() -> HsdAnalysis:
    return HsdAnalysis(p_hate=0.0, labels={}, hs_labels=[], severity="none", severity_score=0.0)


def _active_labels(labels: Dict[str, float], threshold: float) -> List[str]:
    active = [name for name, prob in labels.items() if prob >= threshold]
    return sorted(active, key=lambda n: labels[n], reverse=True)


def _derive_severity(labels: Dict[str, float], p_hate: float) -> tuple[str, float]:
    """Map Detoxify head probs to a severity tier and numeric score."""
    severe = float(labels.get("severe_toxicity", 0.0))
    threat = float(labels.get("threat", 0.0))
    severity_score = max(severe, threat, p_hate)

    if severity_score >= 0.7 or severe >= 0.5 or threat >= 0.5:
        return "severe", severity_score
    if severity_score >= 0.5 or p_hate >= 0.5:
        return "moderate", severity_score
    if severity_score >= 0.2 or p_hate >= 0.2:
        return "mild", severity_score
    return "none", severity_score


def _label_index(id2label: dict, score_label: str) -> int:
    target = score_label.lower()
    for idx, lab in id2label.items():
        if str(lab).lower() == target:
            return int(idx)
    for idx, lab in id2label.items():
        if target in str(lab).lower():
            return int(idx)
    raise ValueError(f"score_label {score_label!r} not found in id2label {id2label!r}")


def _labels_from_logits(logits, model_config, torch) -> Dict[str, float]:
    problem = getattr(model_config, "problem_type", None) or ""
    id2label = {int(k): str(v).lower() for k, v in model_config.id2label.items()}

    if problem == "multi_label_classification":
        probs = torch.sigmoid(logits)
        return {id2label[i]: float(probs[i]) for i in sorted(id2label)}

    probs = torch.softmax(logits, dim=-1)
    return {id2label[i]: float(probs[i]) for i in sorted(id2label)}


def _primary_score(labels: Dict[str, float], score_label: str) -> float:
    if score_label in labels:
        return float(labels[score_label])
    for name, prob in labels.items():
        if score_label.lower() in name:
            return float(prob)
    hate_probs = [prob for name, prob in labels.items() if any(k in name for k in _HATE_KEYS)]
    return max(hate_probs) if hate_probs else 0.0


def analyze_from_logits(
    logits,
    model_config,
    score_label: str,
    label_threshold: float,
    torch,
) -> HsdAnalysis:
    if logits is None or len(logits) == 0:
        return _empty_analysis()
    labels = _labels_from_logits(logits, model_config, torch)
    p_hate = _primary_score(labels, score_label)
    hs_labels = _active_labels(labels, label_threshold)
    severity, severity_score = _derive_severity(labels, p_hate)
    return HsdAnalysis(
        p_hate=p_hate,
        labels=labels,
        hs_labels=hs_labels,
        severity=severity,
        severity_score=severity_score,
    )


class ProxyHateScorer:
    """Fast lexicon/keyword proxy returning P_hate in [0, 1]."""

    name = "proxy"

    def __init__(self, terms: Sequence[str] = (), label_threshold: float = 0.5):
        self._terms = {_skeleton(t) for t in terms if t.strip()}
        self._terms = {s for s in self._terms if len(s) >= 3}
        self._label_threshold = label_threshold

    def analyze(self, text: str) -> HsdAnalysis:
        if self._hit_terms(text) or self._hit_keywords(text):
            labels = {"toxicity": 0.85, "insult": 0.72, "obscene": 0.15}
            hs_labels = _active_labels(labels, self._label_threshold)
            severity, severity_score = _derive_severity(labels, labels["toxicity"])
            return HsdAnalysis(
                p_hate=labels["toxicity"],
                labels=labels,
                hs_labels=hs_labels,
                severity=severity,
                severity_score=severity_score,
            )
        labels = {"toxicity": 0.05}
        return HsdAnalysis(
            p_hate=0.05,
            labels=labels,
            hs_labels=[],
            severity="none",
            severity_score=0.05,
        )

    def score(self, text: str) -> float:
        return self.analyze(text).p_hate

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


class HFToxicityScorer:
    """Open-weight Detoxify-style toxicity classifier."""

    def __init__(
        self,
        model_name: str,
        score_label: str = "toxicity",
        label_threshold: float = 0.5,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self.name = model_name
        self._score_label = score_label
        self._label_threshold = label_threshold
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def analyze(self, text: str) -> HsdAnalysis:
        if not (text or "").strip():
            return _empty_analysis()
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self.model(**enc).logits[0]
        return analyze_from_logits(
            logits,
            self.model.config,
            self._score_label,
            self._label_threshold,
            self._torch,
        )

    def score(self, text: str) -> float:
        return self.analyze(text).p_hate


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


def make_scorer(config: EmHsdConfig) -> object:
    util = config.utility
    if util.backend == "hf":
        return HFToxicityScorer(
            util.model,
            score_label=util.score_label,
            label_threshold=util.label_threshold,
        )
    terms: List[str] = []
    lex = config.spine.lexicon
    if lex.source == "test":
        terms = list(lex.test_terms)
    return ProxyHateScorer(terms, label_threshold=util.label_threshold)


def get_scorer(config: EmHsdConfig) -> object:
    key = f"_scorer_{config.utility.backend}_{config.utility.model}"
    cached = getattr(config, key, None)
    if cached is not None:
        return cached
    scorer = make_scorer(config)
    setattr(config, key, scorer)
    return scorer
