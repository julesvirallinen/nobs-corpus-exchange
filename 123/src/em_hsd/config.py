"""Backward-compatible shim re-exporting the canonical config loader."""

from __future__ import annotations

from em_hsd.core.config import (
    EmbeddingSettings,
    EmHsdConfig,
    EmHsdV2Settings,
    GenerationSettings,
    TriageDpSettings,
    UtilitySettings,
    load_em_hsd_config,
    resolve_config_path,
)

__all__ = [
    "EmHsdConfig",
    "EmHsdV2Settings",
    "GenerationSettings",
    "EmbeddingSettings",
    "UtilitySettings",
    "TriageDpSettings",
    "load_em_hsd_config",
    "resolve_config_path",
]
