"""No-download policy gate for EM-HSD development."""

from __future__ import annotations

import os


class DownloadPolicy:
    """Central gate controlling whether model/data downloads are allowed.

    Default is ``False`` unless the environment variable
    ``EM_HSD_ALLOW_DOWNLOADS`` is set to ``"1"``. This keeps accidental model
    downloads out of CI, smoke tests, and local development.
    """

    _OVERRIDE_ENV_VAR: str = "EM_HSD_ALLOW_DOWNLOADS"

    @classmethod
    def is_allowed(cls) -> bool:
        """Return True only when the caller has explicitly opted in."""
        return os.environ.get(cls._OVERRIDE_ENV_VAR, "") == "1"

    @classmethod
    def require_allowed(cls, action: str) -> None:
        """Raise RuntimeError if downloads are not currently allowed."""
        if not cls.is_allowed():
            raise RuntimeError(
                f"Downloads are disabled for this run. To {action}, set "
                f"{cls._OVERRIDE_ENV_VAR}=1 in the environment."
            )

    @classmethod
    def allowed_message(cls) -> str:
        return (
            f"Downloads are currently {'enabled' if cls.is_allowed() else 'disabled'} "
            f"({cls._OVERRIDE_ENV_VAR}={os.environ.get(cls._OVERRIDE_ENV_VAR, 'unset')})."
        )
