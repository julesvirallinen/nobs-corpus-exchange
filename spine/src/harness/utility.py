"""Utility probe: is HS-classification preserved after privatisation?

For each classifier we report, on original vs privatised Text:
  * F1 vs the gold HS labels (macro), and
  * agreement = fraction of rows where the privatised prediction matches the
    original prediction (label stability under privatisation).

The ensemble ``Utility`` used by the trade-off estimate is the macro-F1 vs gold
averaged over classifiers. Higher is better (classification still works).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import f1_score


def _macro_f1(gold: Sequence[int], pred: Sequence[int]) -> float:
    return float(f1_score(gold, pred, average="macro", zero_division=0))


def utility_report(classifiers: List[object], original_texts: Sequence[str],
                   privatized_texts: Sequence[str],
                   gold_hs: Sequence[int]) -> Dict:
    gold = [int(x) for x in gold_hs]
    per_classifier = []
    f1_orig_list, f1_priv_list = [], []
    for clf in classifiers:
        pred_o = clf.predict(original_texts)
        pred_p = clf.predict(privatized_texts)
        f1_o = _macro_f1(gold, pred_o)
        f1_p = _macro_f1(gold, pred_p)
        agreement = float(np.mean(np.asarray(pred_o) == np.asarray(pred_p)))
        per_classifier.append({
            "classifier": getattr(clf, "name", clf.__class__.__name__),
            "f1_original": f1_o,
            "f1_privatized": f1_p,
            "agreement_priv_vs_orig": agreement,
        })
        f1_orig_list.append(f1_o)
        f1_priv_list.append(f1_p)
    util_orig = float(np.mean(f1_orig_list)) if f1_orig_list else 0.0
    util_priv = float(np.mean(f1_priv_list)) if f1_priv_list else 0.0
    return {
        "per_classifier": per_classifier,
        "utility_original": util_orig,      # ensemble macro-F1 on original
        "utility_privatized": util_priv,    # ensemble macro-F1 on privatised
    }
