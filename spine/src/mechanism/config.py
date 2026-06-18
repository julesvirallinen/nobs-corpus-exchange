"""Configuration for the SPINE mechanism.

The config is loaded from YAML (see configs/*.yaml) into typed dataclasses.
Two fields are *runtime-only* and never come from YAML:

* ``rng``            -- the per-row ``numpy`` Generator (set by the wrapper,
                        one independent generator per row; see SEED_POLICY.md).
* ``uniform_budget`` -- set by the wrapper for ``dpmlm`` mode, which disables the
                        protection step and spends one uniform epsilon on every
                        content token.

The mechanism package intentionally knows nothing about CSV columns, writers,
or row identities beyond the opaque RNG handed to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import yaml

# Sentinel: epsilon value meaning "do not DP-rewrite this token class".
SKIP = None
EpsilonValue = Optional[float]  # float -> spend that epsilon; None -> skip


def _parse_epsilon(value: Union[str, float, int, None]) -> EpsilonValue:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() == "skip":
            return None
        return float(value)
    return float(value)


@dataclass
class MLMConfig:
    backend: str = "hash"            # "hf" | "hash" | "embedding"
    model: str = "distilroberta-base"
    top_k: int = 48
    clip: float = 5.0                # logit clip magnitude C; scores in [-C, C]
    include_original: bool = True
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    candidate_vocab: str = "data/vocab/epsilon1_candidates.txt"
    hybrid_hf: bool = False          # union HF MLM candidates into embedding pool


@dataclass
class EpsilonConfig:
    protected: EpsilonValue = None
    function_word: EpsilonValue = None
    content: EpsilonValue = 6.0
    default: EpsilonValue = 6.0


@dataclass
class NormalizationConfig:
    lowercase: bool = True
    collapse_whitespace: bool = True
    normalize_punctuation: bool = True
    strip_emoji: bool = True
    repair_elongation: bool = True
    fix_misspellings: bool = True


@dataclass
class LexiconConfig:
    source: str = "real"             # "real" | "test"
    path: str = "data/lexicons/hate_terms.txt"
    test_terms: list = field(default_factory=list)
    max_inter_char_gap: int = 2


@dataclass
class SaliencyConfig:
    enabled: bool = False
    model: str = "cardiffnlp/twitter-roberta-base-hate-latest"
    threshold: float = 0.15


@dataclass
class StretchConfig:
    enabled: bool = False
    hard_row_min_tokens: int = 40


@dataclass
class Config:
    mlm: MLMConfig = field(default_factory=MLMConfig)
    epsilon: EpsilonConfig = field(default_factory=EpsilonConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    lexicon: LexiconConfig = field(default_factory=LexiconConfig)
    saliency: SaliencyConfig = field(default_factory=SaliencyConfig)
    stretch: StretchConfig = field(default_factory=StretchConfig)
    protection_enabled: bool = True

    # runtime-only (never serialised)
    rng: object = None               # numpy.random.Generator, set per row
    uniform_budget: bool = False     # dpmlm mode

    def epsilon_for(self, token_class: str) -> EpsilonValue:
        """Resolve the epsilon for a token class, honouring uniform (dpmlm) mode."""
        if self.uniform_budget:
            # dpmlm: one uniform budget on everything that gets rewritten.
            return self.epsilon.content
        return {
            "protected": self.epsilon.protected,
            "function_word": self.epsilon.function_word,
            "content": self.epsilon.content,
        }.get(token_class, self.epsilon.default)


def config_from_dict(d: dict) -> Config:
    d = d or {}
    mlm = d.get("mlm", {}) or {}
    eps = d.get("epsilon", {}) or {}
    norm = d.get("normalization", {}) or {}
    lex = d.get("lexicon", {}) or {}
    sal = d.get("saliency", {}) or {}
    stretch = d.get("stretch", {}) or {}
    protection = d.get("protection", {}) or {}

    return Config(
        mlm=MLMConfig(
            backend=str(mlm.get("backend", "hash")),
            model=str(mlm.get("model", "distilroberta-base")),
            top_k=int(mlm.get("top_k", 48)),
            clip=float(mlm.get("clip", 5.0)),
            include_original=bool(mlm.get("include_original", True)),
            embedding_model=str(
                mlm.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            candidate_vocab=str(
                mlm.get("candidate_vocab", "data/vocab/epsilon1_candidates.txt")
            ),
            hybrid_hf=bool(mlm.get("hybrid_hf", False)),
        ),
        epsilon=EpsilonConfig(
            protected=_parse_epsilon(eps.get("protected", "skip")),
            function_word=_parse_epsilon(eps.get("function_word", "skip")),
            content=_parse_epsilon(eps.get("content", 6.0)),
            default=_parse_epsilon(eps.get("default", 6.0)),
        ),
        normalization=NormalizationConfig(
            lowercase=bool(norm.get("lowercase", True)),
            collapse_whitespace=bool(norm.get("collapse_whitespace", True)),
            normalize_punctuation=bool(norm.get("normalize_punctuation", True)),
            strip_emoji=bool(norm.get("strip_emoji", True)),
            repair_elongation=bool(norm.get("repair_elongation", True)),
            fix_misspellings=bool(norm.get("fix_misspellings", True)),
        ),
        lexicon=LexiconConfig(
            source=str(lex.get("source", "real")),
            path=str(lex.get("path", "data/lexicons/hate_terms.txt")),
            test_terms=list(lex.get("test_terms", []) or []),
            max_inter_char_gap=int(lex.get("max_inter_char_gap", 2)),
        ),
        saliency=SaliencyConfig(
            enabled=bool(sal.get("enabled", False)),
            model=str(sal.get("model", "cardiffnlp/twitter-roberta-base-hate-latest")),
            threshold=float(sal.get("threshold", 0.15)),
        ),
        stretch=StretchConfig(
            enabled=bool(stretch.get("enabled", False)),
            hard_row_min_tokens=int(stretch.get("hard_row_min_tokens", 40)),
        ),
        protection_enabled=bool(protection.get("enabled", True)),
    )


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return config_from_dict(data)
