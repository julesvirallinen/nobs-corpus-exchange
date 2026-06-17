"""EM-HSD 2.0 orchestrator: Phase 1 (ε₁ + k paraphrases) + Phase 2 (filter + EM)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .config import EmHsdConfig
from .constraints import filter_candidates
from .dp_select import select_rewrite
from .generative_proposer import get_proposer
from .prune_candidates import prune_candidates
from .resources import protected_canonicals
from .sensitivity import selection_sensitivity
from .token_sanitize import token_sanitize
from .utility_scorer import get_scorer


def privatize_em_hsd_v2(text: str, config: EmHsdConfig) -> Tuple[str, Dict[str, Any]]:
    """Privatise one Text string. Returns (output, audit_dict)."""
    if config.spine.rng is None:
        raise ValueError(
            "config.spine.rng must be set before privatize_em_hsd_v2"
        )

    em = config.em_hsd_v2
    epsilon_1 = em.epsilon_1
    epsilon_2 = em.epsilon_2
    delta_u = selection_sensitivity(text, em.use_refined_delta_u)

    x_priv, token_log = token_sanitize(text, config, epsilon_1)
    canonicals, skels = protected_canonicals(text, config)

    audit: Dict[str, Any] = {
        "mode": "em-hsd-v2",
        "epsilon_total": em.epsilon_total,
        "epsilon_1": epsilon_1,
        "epsilon_2": epsilon_2,
        "delta_u": delta_u,
        "delta_u_naive": 1.0,
        "x_priv": x_priv,
        "protected_terms": canonicals,
        "token_log": token_log,
        "fallback": False,
        "fallback_reason": "",
        "k_generated": 0,
        "k_after_prune": 0,
        "k_valid": 0,
        "candidates": [],
        "filter_details": [],
    }

    proposer = get_proposer(config)
    proposer.bind(config.spine.rng, canonicals)
    scorer = get_scorer(config)
    p_orig = float(scorer.score(text))
    p_x_priv = float(scorer.score(x_priv))
    audit["P_hate_original"] = p_orig
    audit["P_hate_x_priv"] = p_x_priv
    audit["utility_backend"] = config.utility.backend
    audit["utility_model"] = getattr(scorer, "name", config.utility.model)

    from .embedding import get_encoder
    encoder = get_encoder(config)

    try:
        raw_candidates = proposer.propose(x_priv, em.k_generate)
    except Exception as exc:
        audit["fallback"] = True
        audit["fallback_reason"] = f"proposer_error:{exc}"
        return x_priv, audit

    audit["k_generated"] = len(raw_candidates)
    pruned = prune_candidates(raw_candidates, config)
    audit["k_after_prune"] = len(pruned)

    batch = filter_candidates(
        pruned, text, x_priv, skels, config, scorer, encoder,
    )
    audit["filter_details"] = [
        {
            "text": d.candidate[:200],
            "valid": d.valid,
            "reject": d.reject,
            "p_hate": d.p_hate,
            "sem_cos": d.sem_cos,
        }
        for d in batch.details
    ]

    valid = batch.valid
    scores = batch.scores
    audit["k_valid"] = len(valid)

    if len(valid) >= 2:
        chosen, sel = select_rewrite(
            valid, scores, epsilon_2, clip=1.0,
            rng=config.spine.rng, sensitivity=delta_u,
        )
        audit["candidates"] = [
            {"text": c[:200], "score": s, "selected": c == chosen}
            for c, s in zip(valid, scores)
        ]
        audit["selection_probs"] = sel.probs.tolist()
        return chosen, audit

    if len(valid) == 1:
        audit["candidates"] = [{"text": valid[0][:200], "score": scores[0], "selected": True}]
        audit["selection_probs"] = [1.0]
        return valid[0], audit

    audit["fallback"] = True
    audit["fallback_reason"] = "no_valid_candidates"
    return x_priv, audit
