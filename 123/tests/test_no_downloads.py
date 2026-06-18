from __future__ import annotations

import sys
from unittest.mock import patch

from mechanism.rng import make_row_rng

from em_hsd import load_em_hsd_config, privatize_em_hsd_v2


def test_no_download_functions_called_during_mock_run():
    """Patch all direct download entry points and run the mock pipeline."""
    cfg = load_em_hsd_config("configs/em-hsd-v2-test.yaml")
    cfg.spine.rng = make_row_rng(40, run_seed="test")

    fake_transformers = type(sys)("transformers")
    fake_transformers.AutoModelForCausalLM = type(sys)("AutoModelForCausalLM")
    fake_transformers.AutoModelForCausalLM.from_pretrained = lambda *a, **k: None
    fake_transformers.AutoModelForSequenceClassification = type(sys)("AutoModelForSequenceClassification")
    fake_transformers.AutoModelForSequenceClassification.from_pretrained = lambda *a, **k: None
    fake_transformers.AutoTokenizer = type(sys)("AutoTokenizer")
    fake_transformers.AutoTokenizer.from_pretrained = lambda *a, **k: None
    fake_unsloth = type(sys)("unsloth")
    fake_unsloth.FastLanguageModel = type(sys)("FastLanguageModel")
    fake_unsloth.FastLanguageModel.from_pretrained = lambda *a, **k: (None, None)
    fake_st = type(sys)("sentence_transformers")
    fake_st.SentenceTransformer = lambda *a, **k: None

    with (
        patch.dict(
            sys.modules,
            {
                "transformers": fake_transformers,
                "unsloth": fake_unsloth,
                "sentence_transformers": fake_st,
            },
        ),
        patch.object(fake_transformers.AutoModelForCausalLM, "from_pretrained") as mock_causal,
        patch.object(
            fake_transformers.AutoModelForSequenceClassification, "from_pretrained"
        ) as mock_seq,
        patch.object(fake_transformers.AutoTokenizer, "from_pretrained") as mock_tok,
        patch.object(fake_unsloth.FastLanguageModel, "from_pretrained") as mock_unsloth,
        patch.object(fake_st, "SentenceTransformer") as mock_st,
    ):
        out, audit = privatize_em_hsd_v2("you are a dummy", cfg)
        assert isinstance(out, str) and out.strip()
        mock_causal.assert_not_called()
        mock_seq.assert_not_called()
        mock_tok.assert_not_called()
        mock_unsloth.assert_not_called()
        mock_st.assert_not_called()


def test_download_policy_defaults_to_false():
    from em_hsd.core.policy import DownloadPolicy

    assert DownloadPolicy.is_allowed() is False
