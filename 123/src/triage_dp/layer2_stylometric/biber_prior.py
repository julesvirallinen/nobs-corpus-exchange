"""Layer 2 — Biber-style stylometric prior.

Layer 1 occlusion can miss *systematic grammatical* identity carriers because
removing one such token barely moves the hate classifier. Biber's register
framework flags these classes as high authorship-identity / low hate-utility:
2nd-person pronouns, place & time adverbials, and discourse particles
(Arnold et al. show DP rewriting is register-blind to exactly these).

This prior scans the text for those classes and emits ``Q2`` routes with a
``biber_boost`` and a ``biber:<class>`` reason, so Layer 4 ε₁-rewrites them even
though they are closed-class function words it would otherwise keep. It never
overrides a Layer 1 protect route (hate signal wins).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from em_hsd.core.config import EmHsdConfig
from em_hsd.interfaces.triage import TokenRoute

# Biber identity-carrier classes (online-hate relevance from the Layer 2 spec).
_SECOND_PERSON = {"you", "your", "yours", "youre", "yourself", "yall", "u", "ur"}
_PLACE_ADV = {"here", "there", "downtown", "abroad", "home", "nearby", "uptown"}
_TIME_ADV = {"now", "today", "tonight", "tomorrow", "yesterday", "currently", "lately"}
_DISCOURSE = {
    "honestly", "literally", "basically", "actually", "frankly", "seriously",
    "lol", "lmao", "imo", "tbh", "fr", "ngl",
}
_CLASSES = (
    ("2nd-person", _SECOND_PERSON, 0.6),
    ("place-adverbial", _PLACE_ADV, 0.4),
    ("time-adverbial", _TIME_ADV, 0.4),
    ("discourse-particle", _DISCOURSE, 0.5),
)
_WORD = re.compile(r"\S+")


def _key(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


class BiberStylometricPrior:
    """Boost privacy treatment of register-level identity carriers."""

    def boost(
        self,
        text: str,
        token_routes: Sequence[TokenRoute],
        config: EmHsdConfig,
    ) -> list[TokenRoute]:
        routes = list(token_routes)
        # Tokens already routed by Layer 1 keep their decision.
        claimed = {r.token for r in routes}
        boosts = (getattr(config.triage_dp, "layer2", {}) or {}).get("biber_boosts", {})
        for m in _WORD.finditer(text):
            tok = m.group()
            if tok in claimed:
                continue
            key = _key(tok)
            for name, members, default_boost in _CLASSES:
                if key in members:
                    boost = float(boosts.get(name, default_boost))
                    routes.append(
                        TokenRoute(
                            token=tok,
                            start=m.start(),
                            end=m.end(),
                            quadrant="Q2",
                            action="sanitize",
                            biber_boost=boost,
                            reason=f"biber:{name} (stylometric identity carrier)",
                        )
                    )
                    claimed.add(tok)
                    break
        return routes


__all__ = ["BiberStylometricPrior"]
