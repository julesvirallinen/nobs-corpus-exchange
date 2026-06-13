"""The local trade-off estimate.

    TO = (Utility_privatized / Utility_original)
       - (Privacy_privatized / Privacy_original)

where Utility is ensemble HS-classification macro-F1 (higher = better) and
Privacy is the re-identification attacker's top-1 accuracy (LOWER = better, so a
SMALLER privacy ratio is better and RAISES TO).

IMPORTANT: TO is a LOCAL APPROXIMATION of the organiser's hidden evaluator, not
the official score. It is computed from our own probe models and must not be
overfit to. See README "Evaluation honesty".
"""

from __future__ import annotations

from typing import Dict

LABEL = "LOCAL APPROXIMATION (not the official score)"


def _ratio(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def trade_off(utility_original: float, utility_privatized: float,
              privacy_original: float, privacy_privatized: float) -> Dict:
    util_ratio = _ratio(utility_privatized, utility_original)
    priv_ratio = _ratio(privacy_privatized, privacy_original)
    return {
        "label": LABEL,
        "utility_ratio": util_ratio,
        "privacy_ratio": priv_ratio,
        "trade_off_estimate": util_ratio - priv_ratio,
    }
