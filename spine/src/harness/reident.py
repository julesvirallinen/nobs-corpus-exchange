"""Re-identification (stylometry) probes — attacker models.

Probe A (always on): character n-gram TF-IDF + linear SVM authorship classifier,
trained on the ORIGINAL Text with Author labels, then asked to re-identify the
author of each PRIVATISED Text. Lower privatised accuracy = better privacy.

Probe B (optional, --reident-transformer, slower, HF-gated): a frozen
transformer linear-probe — mean-pooled embeddings + logistic regression — as a
stronger, learned stylometric attacker. (Full fine-tuning is heavier; this is the
bounded, documented version.)

Both report top-1 accuracy on original (an upper bound / sanity check) vs
privatised. This is the only module besides evaluate.py that uses Author labels.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC


def char_ngram_reident(original_texts: Sequence[str],
                       privatized_texts: Sequence[str],
                       authors: Sequence[str]) -> Dict:
    authors = list(authors)
    n_authors = len(set(authors))
    if n_authors < 2:
        return {"probe": "char-ngram-svm", "n_authors": n_authors,
                "accuracy_original": 0.0, "accuracy_privatized": 0.0,
                "note": "need >=2 authors for a meaningful probe"}

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    x_orig = vectorizer.fit_transform(original_texts)
    clf = LinearSVC(random_state=0)
    clf.fit(x_orig, authors)

    acc_orig = float(accuracy_score(authors, clf.predict(x_orig)))
    x_priv = vectorizer.transform(privatized_texts)
    acc_priv = float(accuracy_score(authors, clf.predict(x_priv)))
    return {
        "probe": "char-ngram-svm",
        "n_authors": n_authors,
        "accuracy_original": acc_orig,      # train-set upper bound / sanity
        "accuracy_privatized": acc_priv,    # the attacker's success after privatising
    }


def transformer_reident(original_texts: Sequence[str],
                        privatized_texts: Sequence[str],
                        authors: Sequence[str],
                        model_name: str = "distilroberta-base") -> Dict:
    """Optional, slower, HF-gated frozen-feature linear-probe attacker."""
    import torch
    from sklearn.linear_model import LogisticRegression
    from transformers import AutoModel, AutoTokenizer

    authors = list(authors)
    if len(set(authors)) < 2:
        return {"probe": "transformer-linear-probe", "accuracy_original": 0.0,
                "accuracy_privatized": 0.0, "note": "need >=2 authors"}

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            enc = tok(t or " ", return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state[0]
            mask = enc["attention_mask"][0].unsqueeze(-1)
            pooled = (hidden * mask).sum(0) / mask.sum().clamp(min=1)
            vecs.append(pooled.numpy())
        return np.vstack(vecs)

    emb_o = embed(original_texts)
    emb_p = embed(privatized_texts)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(emb_o, authors)
    return {
        "probe": "transformer-linear-probe",
        "model": model_name,
        "accuracy_original": float(accuracy_score(authors, clf.predict(emb_o))),
        "accuracy_privatized": float(accuracy_score(authors, clf.predict(emb_p))),
    }


def reident_report(original_texts: Sequence[str], privatized_texts: Sequence[str],
                   authors: Sequence[str], use_transformer: bool = False,
                   transformer_model: str = "distilroberta-base") -> Dict:
    probes = [char_ngram_reident(original_texts, privatized_texts, authors)]
    if use_transformer:
        try:
            probes.append(transformer_reident(original_texts, privatized_texts,
                                              authors, transformer_model))
        except Exception as exc:  # pragma: no cover
            probes.append({"probe": "transformer-linear-probe", "error": str(exc)})
    primary = probes[0]
    return {
        "probes": probes,
        # privacy metric = primary attacker's top-1 accuracy (lower is better)
        "privacy_original": primary["accuracy_original"],
        "privacy_privatized": primary["accuracy_privatized"],
    }
