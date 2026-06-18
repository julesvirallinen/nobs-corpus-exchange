"""Cached resource helpers for EM-HSD pipeline."""

from __future__ import annotations

from mechanism.spine import get_resources

from em_hsd.core.config import EmHsdConfig
from em_hsd.core.policy import DownloadPolicy
from em_hsd.layer4.filter import protected_skeletons


def protected_canonicals(text: str, config: EmHsdConfig) -> tuple[list[str], list[str]]:
    """Return (canonical terms, skeletons) found in text."""
    lexicon, _, _ = get_resources(config.spine)
    if not lexicon or not lexicon.loaded:
        return [], []
    spans = lexicon.find_protected_spans(text)
    canonicals = sorted({canon for _s, _e, canon in spans})
    return canonicals, protected_skeletons(canonicals)


def init_spine_resources(config: EmHsdConfig) -> None:
    """Fail early if SPINE backends cannot load."""
    get_resources(config.spine)


class ResourceManager:
    """Lazy singletons for scorer, encoder, proposer and MLM backend.

    Guards model downloads unless ``EM_HSD_ALLOW_DOWNLOADS=1`` is set or
    ``allow_downloads`` is explicitly passed.
    """

    def __init__(self, config: EmHsdConfig, *, allow_downloads: bool | None = None):
        self.config = config
        self.allow_downloads = (
            DownloadPolicy.is_allowed() if allow_downloads is None else allow_downloads
        )
        self._scorer: object = None
        self._encoder: object = None
        self._proposer: object = None
        self._mlm_backend: object = None

    def _guard_download(self, action: str) -> None:
        if not self.allow_downloads:
            raise RuntimeError(
                f"Downloads are disabled for this run. To {action}, set "
                f"{DownloadPolicy._OVERRIDE_ENV_VAR}=1 in the environment."
            )

    def scorer(self) -> object:
        if self._scorer is not None:
            return self._scorer
        from em_hsd.layer4.scorer import make_scorer
        if self.config.utility.backend == "hf":
            self._guard_download("load hate utility model from Hugging Face")
        self._scorer = make_scorer(self.config)
        return self._scorer

    def encoder(self) -> object:
        if self._encoder is not None:
            return self._encoder
        from em_hsd.core.embedding import make_encoder
        if self.config.embedding.backend in ("hf", "auto"):
            self._guard_download("load sentence transformer model")
        self._encoder = make_encoder(self.config.embedding)
        return self._encoder

    def proposer(self) -> object:
        if self._proposer is not None:
            return self._proposer
        from em_hsd.layer4.proposer import make_proposer
        if self.config.generation.backend in ("unsloth", "transformers"):
            self._guard_download(f"load generative model ({self.config.generation.backend})")
        self._proposer = make_proposer(self.config)
        return self._proposer

    def mlm_backend(self) -> object:
        if self._mlm_backend is not None:
            return self._mlm_backend
        from mechanism.spine import get_resources
        self._mlm_backend = get_resources(self.config.spine)[1]
        return self._mlm_backend
