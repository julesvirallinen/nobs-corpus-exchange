"""EM-HSD 2.0 orchestrator: Phase 1 (ε₁ + k paraphrases) + Phase 2 (filter + EM)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import EmHsdConfig
from .candidate_sanitize import drop_prompt_echoes
from .constraints import filter_candidates, normalized_edit_ratio
from .dp_select import select_rewrite
from .generative_proposer import get_proposer
from .prune_candidates import prune_candidates
from .resources import merge_protected_terms, protected_canonicals, protected_from_token_log
from .sensitivity import selection_sensitivity
from .token_sanitize import token_sanitize
from .utility_scorer import get_scorer


def privatize_em_hsd_v2(
    text: str,
    config: EmHsdConfig,
    *,
    original_text: Optional[str] = None,
    protected_spans: Optional[Sequence[str]] = None,
    upstream_token_log: Optional[Sequence[Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Privatise one Text string. Returns (output, audit_dict).

    Standalone (default): ``text`` is raw ``x``; runs full ε₁ + paraphrase + EM.

    Composed (TRIAGE-DP L1–L4): ``text`` is token-sanitized ``T′``;
    pass ``original_text=x`` for hate/semantic floors; optional
    ``protected_spans`` from Layer 1 audit; ``upstream_token_log`` merged into audit.
    """
    if config.spine.rng is None:
        raise ValueError(
            "config.spine.rng must be set before privatize_em_hsd_v2"
        )

    em = config.em_hsd_v2
    reference = original_text if original_text is not None else text
    epsilon_1 = em.epsilon_1
    epsilon_2 = em.epsilon_2
    delta_u = selection_sensitivity(reference, em.use_refined_delta_u)

    if em.skip_phase_1a:
        x_priv = text
        token_log: List[Any] = list(upstream_token_log or [])
    else:
        x_priv, token_log = token_sanitize(text, config, epsilon_1)
        if upstream_token_log:
            token_log = list(upstream_token_log) + list(token_log)

    if protected_spans is not None:
        lex_canon = sorted({c for c in protected_spans if c and str(c).strip()})
    else:
        lex_canon, _ = protected_canonicals(reference, config)

    sal_canon = protected_from_token_log(token_log)
    canonicals, skels = merge_protected_terms(lex_canon, sal_canon)

    scorer = get_scorer(config)
    hsd_orig = scorer.analyze(reference)
    hsd_x_priv = scorer.analyze(x_priv)

    audit: Dict[str, Any] = {
        "mode": "em-hsd-v2",
        "deployment_mode": em.deployment_mode,
        "epsilon_total": em.epsilon_total,
        "epsilon_1": epsilon_1,
        "epsilon_2": epsilon_2,
        "epsilon_1_spent_here": epsilon_1,
        "delta_u": delta_u,
        "delta_u_naive": 1.0,
        "x_priv": x_priv,
        "reference_text": reference[:500] if reference != text else None,
        "protected_terms": canonicals,
        "protected_terms_lexicon": lex_canon,
        "protected_terms_saliency": sal_canon,
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
    proposer.bind(config.spine.rng, canonicals, p_hate_original=hsd_orig.p_hate)
    audit["P_hate_original"] = hsd_orig.p_hate
    audit["P_hate_x_priv"] = hsd_x_priv.p_hate
    audit["hsd_original"] = hsd_orig.to_dict()
    audit["hsd_x_priv"] = hsd_x_priv.to_dict()
    audit["utility_backend"] = config.utility.backend
    audit["utility_model"] = getattr(scorer, "name", config.utility.model)

    from .embedding import get_encoder
    encoder = get_encoder(config)

    try:
        raw_candidates = proposer.propose(x_priv, em.k_generate)
    except Exception as exc:
        audit["fallback"] = True
        audit["fallback_reason"] = f"proposer_error:{exc}"
        audit["hsd_output"] = hsd_x_priv.to_dict()
        return x_priv, audit

    audit["prompt_profile"] = getattr(proposer, "last_prompt_profile", None)
    audit["prompt_variants_used"] = getattr(proposer, "last_prompt_variants", [])

    audit["k_generated"] = len(raw_candidates)
    pruned = prune_candidates(raw_candidates, config)
    audit["k_after_prune"] = len(pruned)
    echoed, dropped_echoes = drop_prompt_echoes(pruned)
    audit["k_after_echo_drop"] = len(echoed)
    audit["echo_dropped"] = dropped_echoes

    batch = filter_candidates(
        echoed,
        reference,
        x_priv,
        skels,
        config,
        scorer,
        encoder,
        protected_terms=canonicals,
        p_hate_x_priv=hsd_x_priv.p_hate,
    )
    audit["filter_details"] = [
        {
            "text": d.candidate[:200],
            "valid": d.valid,
            "reject": d.reject,
            "p_hate": d.p_hate,
            "sem_cos": d.sem_cos,
            "sem_cos_x_priv": d.sem_cos_x_priv,
            "severity": d.severity,
            "severity_score": d.severity_score,
            "hs_labels": d.hs_labels,
            "labels": d.label_probs,
        }
        for d in batch.details
    ]

    valid = batch.valid
    scores = list(batch.scores)
    audit["k_valid"] = len(valid)

    alpha = em.utility_alpha
    if valid and alpha < 1.0:
        scores = [
            alpha * s + (1.0 - alpha) * min(1.0, normalized_edit_ratio(c, reference))
            for c, s in zip(valid, scores)
        ]
        audit["selection_scores_blended"] = True
    else:
        audit["selection_scores_blended"] = False

    if len(valid) >= 2:
        chosen, sel = select_rewrite(
            valid, scores, epsilon_2, clip=1.0,
            rng=config.spine.rng, sensitivity=delta_u,
        )
        audit["candidates"] = [
            {
                "text": c[:200],
                "score": s,
                "selected": c == chosen,
                **next(
                    (
                        {
                            "severity": d.severity,
                            "severity_score": d.severity_score,
                            "hs_labels": d.hs_labels,
                            "labels": d.label_probs,
                        }
                        for d in batch.details
                        if d.valid and d.candidate == c
                    ),
                    {},
                ),
            }
            for c, s in zip(valid, scores)
        ]
        audit["selection_probs"] = sel.probs.tolist()
        audit["hsd_output"] = scorer.analyze(chosen).to_dict()
        return chosen, audit

    if len(valid) == 1:
        d0 = next(d for d in batch.details if d.valid)
        audit["candidates"] = [{
            "text": valid[0][:200],
            "score": scores[0],
            "selected": True,
            "severity": d0.severity,
            "severity_score": d0.severity_score,
            "hs_labels": d0.hs_labels,
            "labels": d0.label_probs,
        }]
        audit["selection_probs"] = [1.0]
        audit["hsd_output"] = hsd_x_priv.to_dict() if valid[0] == x_priv else scorer.analyze(valid[0]).to_dict()
        return valid[0], audit

    audit["fallback"] = True
    audit["fallback_reason"] = "no_valid_candidates"
    audit["hsd_output"] = hsd_x_priv.to_dict()
    return x_priv, audit
