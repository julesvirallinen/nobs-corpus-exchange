"""Backward-compatible shim for legacy `em_hsd.resources` imports."""

from __future__ import annotations

from em_hsd.core.resources import ResourceManager, init_spine_resources, protected_canonicals

__all__ = ["ResourceManager", "init_spine_resources", "protected_canonicals"]
