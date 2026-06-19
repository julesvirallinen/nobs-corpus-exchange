"""EM-HSD 2.0 — Layer-4-only privatization pipeline.

Sub-packages:
    core        – primitives (config, DP selection, sensitivity, embedding, resources, policy, paths)
    layer4      – privatisation components (orchestrator, proposer, filter, scorer, pruner)
    interfaces  – public adaptors for TRIAGE-DP integration
    cli         – command-line entry points
    io          – input/output helpers
"""

from __future__ import annotations

from em_hsd.core.config import EmHsdConfig, load_em_hsd_config
from em_hsd.layer4.orchestrator import Layer4Orchestrator, privatize_em_hsd_v2

__all__ = [
    "EmHsdConfig",
    "load_em_hsd_config",
    "Layer4Orchestrator",
    "privatize_em_hsd_v2",
]
