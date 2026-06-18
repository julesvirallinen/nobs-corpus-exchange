from __future__ import annotations

from em_hsd import (
    EmHsdConfig,
    Layer4Orchestrator,
    load_em_hsd_config,
    privatize_em_hsd_v2,
)
from em_hsd.constraints import (
    FilterBatch,
    FilterResult,
    filter_candidates,
    protected_skeletons,
    spans_preserved,
)
from em_hsd.csv_compat import read_csv_compat, write_csv_compat
from em_hsd.dp_select import select_rewrite
from em_hsd.embedding import SimpleEncoder, get_encoder
from em_hsd.generative_proposer import MockProposer, get_proposer
from em_hsd.prune_candidates import prune_candidates
from em_hsd.resources import ResourceManager, protected_canonicals
from em_hsd.sensitivity import refined_delta_u, selection_sensitivity
from em_hsd.token_sanitize import token_sanitize
from em_hsd.utility_scorer import ProxyHateScorer, get_scorer


def test_backward_compat_imports():
    assert callable(privatize_em_hsd_v2)
    assert callable(load_em_hsd_config)
    assert callable(Layer4Orchestrator)
    assert callable(token_sanitize)
    assert callable(select_rewrite)
    assert callable(filter_candidates)
    assert callable(prune_candidates)
    assert callable(get_scorer)
    assert callable(get_proposer)
    assert callable(get_encoder)
    assert callable(read_csv_compat)
    assert callable(write_csv_compat)
    assert callable(protected_canonicals)
    assert callable(refined_delta_u)
    assert callable(selection_sensitivity)
    assert isinstance(ResourceManager, type)
    assert isinstance(FilterResult, type)
    assert isinstance(FilterBatch, type)
    assert isinstance(SimpleEncoder, type)
    assert isinstance(MockProposer, type)
    assert isinstance(ProxyHateScorer, type)
    assert EmHsdConfig is not None
    assert spans_preserved("dummy", protected_skeletons(["dummy"]))
