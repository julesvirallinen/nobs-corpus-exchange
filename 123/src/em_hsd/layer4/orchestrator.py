"""Layer-4 orchestrator: Phase 1 token sanitize + Phase 2 generate/filter/select."""

from __future__ import annotations

from typing import Any

from em_hsd.core.config import EmHsdConfig
from em_hsd.core.dp_select import select_rewrite
from em_hsd.core.embedding import get_encoder
from em_hsd.core.resources import protected_canonicals
from em_hsd.core.sanitize import token_sanitize
from em_hsd.core.sensitivity import selection_sensitivity
from em_hsd.interfaces.triage import TokenRoute
from em_hsd.layer4.filter import filter_candidates
from em_hsd.layer4.proposer import get_proposer
from em_hsd.layer4.prune import prune_candidates
from em_hsd.layer4.scorer import get_scorer


class Layer4Orchestrator:
    """Sentence-level EM-HSD pipeline with explicit fallback policy."""

    @staticmethod
    def _apply_layer1_routes(
        text: str,
        config: EmHsdConfig,
        routes: list[TokenRoute] | None,
    ) -> tuple[str, list[str], list[dict]]:
        """Run token sanitize with optional protected overrides from Layer 1.

        Q1 and Q3 routes set protected_override; Q2 uses configured ε₁.
        """
        epsilon_1 = config.em_hsd_v2.epsilon_1
        protected_override: set[str] = set()
        token_log: list[dict] = []
        if routes:
            for route in routes:
                if route.quadrant in ("Q1", "Q3") or route.protected_override:
                    protected_override.add(route.token)
        x_priv, token_log = token_sanitize(text, config, epsilon_1)
        return x_priv, sorted(protected_override), token_log

    def privatize(
        self,
        text: str,
        config: EmHsdConfig,
        *,
        layer1_routes: list[TokenRoute] | None = None,
        layer2_routes: list[TokenRoute] | None = None,
        layer3_overrides: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        """Privatise one text string.

        Fallback policy:
          - ≥2 valid candidates  → exponential-mechanism select with ε₂ + refined Δu
          - 1 valid candidate      → return it
          - 0 valid / proposer fail → return ``x_priv`` (ε₁ still held)
          - never return the raw input unchanged
        """
        if config.spine.rng is None:
            raise ValueError("config.spine.rng must be set before privatize")

        if layer3_overrides:
            em = config.em_hsd_v2
            for key, value in layer3_overrides.items():
                if hasattr(em, key):
                    setattr(em, key, value)

        em = config.em_hsd_v2
        epsilon_1 = em.epsilon_1
        epsilon_2 = em.epsilon_2
        delta_u = selection_sensitivity(text, em.use_refined_delta_u)

        x_priv, token_log = token_sanitize(text, config, epsilon_1)
        canonicals, skels = protected_canonicals(text, config)

        if layer2_routes is not None:
            pass  # Layer 2 priors are applied upstream; no action in Layer 4

        audit: dict[str, Any] = {
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
        encoder = get_encoder(config)
        score = getattr(scorer, "score")
        p_orig = float(score(text))
        p_x_priv = float(score(x_priv))
        audit["P_hate_original"] = p_orig
        audit["P_hate_x_priv"] = p_x_priv
        audit["utility_backend"] = config.utility.backend
        audit["utility_model"] = getattr(scorer, "name", config.utility.model)

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
                for c, s in zip(valid, scores, strict=True)
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


def privatize_em_hsd_v2(
    text: str,
    config: EmHsdConfig,
    **kwargs,
) -> tuple[str, dict]:
    """Thin wrapper around ``Layer4Orchestrator.privatize``."""
    return Layer4Orchestrator().privatize(text, config, **kwargs)
