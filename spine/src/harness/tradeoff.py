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
