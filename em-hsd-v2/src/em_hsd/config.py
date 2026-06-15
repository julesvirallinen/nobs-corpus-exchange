"""EM-HSD 2.0 configuration (extends SPINE YAML with layer-4 fields)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from . import spine_bootstrap  # noqa: F401 — path setup before mechanism import

from mechanism.config import Config, config_from_dict


@dataclass
class EmHsdV2Settings:
    epsilon_total: float = 18.0
    epsilon_split: float = 0.5
    k_generate: int = 6
    k_max_after_prune: int = 4
    tau_dup: float = 0.80
    token_sanitize_top_m: int = 32
    generation_temperature: float = 0.9
    hate_floor_delta: float = 0.05
    tau_sem_min: float = 0.55
    min_edit_ratio: float = 0.08
    clip: float = 5.0
    use_refined_delta_u: bool = True
    utility_alpha: float = 1.0

    @property
    def epsilon_1(self) -> float:
        return self.epsilon_total * self.epsilon_split

    @property
    def epsilon_2(self) -> float:
        return self.epsilon_total * self.epsilon_split


@dataclass
class GenerationSettings:
    backend: str = "mock"
    model: str = "unsloth/Qwen3.5-0.8B"
    load_in_4bit: bool = True
    max_new_tokens: int = 256


@dataclass
class EmbeddingSettings:
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    backend: str = "auto"


@dataclass
class EmHsdConfig:
    """SPINE mechanism config plus EM-HSD v2 layer settings."""

    spine: Config
    em_hsd_v2: EmHsdV2Settings = field(default_factory=EmHsdV2Settings)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)

    @property
    def rng(self):
        return self.spine.rng

    @rng.setter
    def rng(self, value):
        self.spine.rng = value


def _parse_em_settings(d: dict) -> EmHsdV2Settings:
    em = d.get("em_hsd_v2", {}) or {}
    return EmHsdV2Settings(
        epsilon_total=float(em.get("epsilon_total", 18.0)),
        epsilon_split=float(em.get("epsilon_split", 0.5)),
        k_generate=int(em.get("k_generate", 6)),
        k_max_after_prune=int(em.get("k_max_after_prune", 4)),
        tau_dup=float(em.get("tau_dup", 0.80)),
        token_sanitize_top_m=int(em.get("token_sanitize_top_m", 32)),
        generation_temperature=float(em.get("generation_temperature", 0.9)),
        hate_floor_delta=float(em.get("hate_floor_delta", 0.05)),
        tau_sem_min=float(em.get("tau_sem_min", 0.55)),
        min_edit_ratio=float(em.get("min_edit_ratio", 0.08)),
        clip=float(em.get("clip", 5.0)),
        use_refined_delta_u=bool(em.get("use_refined_delta_u", True)),
        utility_alpha=float(em.get("utility_alpha", 1.0)),
    )


def _parse_generation(d: dict) -> GenerationSettings:
    gen = d.get("generation", {}) or {}
    return GenerationSettings(
        backend=str(gen.get("backend", "mock")),
        model=str(gen.get("model", "unsloth/Qwen3.5-2B")),
        load_in_4bit=bool(gen.get("load_in_4bit", True)),
        max_new_tokens=int(gen.get("max_new_tokens", 256)),
    )


def _parse_embedding(d: dict) -> EmbeddingSettings:
    emb = d.get("embedding", {}) or {}
    return EmbeddingSettings(
        model=str(emb.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        backend=str(emb.get("backend", "auto")),
    )


def load_em_hsd_config(path: str) -> EmHsdConfig:
    cfg_path = Path(path).resolve()
    with open(cfg_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    spine = config_from_dict(data)
    # Resolve lexicon path relative to config file / Johnny tree.
    lex_path = Path(spine.lexicon.path)
    if not lex_path.is_file():
        for base in (cfg_path.parent, cfg_path.parent.parent):
            candidate = (base / spine.lexicon.path).resolve()
            if candidate.is_file():
                spine.lexicon.path = str(candidate)
                break
        else:
            johnny = cfg_path.parent.parent / "Johnny t0-1.03" / "data" / "lexicons" / "hate_terms.txt"
            if johnny.is_file():
                spine.lexicon.path = str(johnny)
    return EmHsdConfig(
        spine=spine,
        em_hsd_v2=_parse_em_settings(data),
        generation=_parse_generation(data),
        embedding=_parse_embedding(data),
    )


def resolve_config_path(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p)
    root = Path(__file__).resolve().parents[2]
    candidate = root / "configs" / path
    if candidate.is_file():
        return str(candidate)
    return path
