from __future__ import annotations

import os
from pathlib import Path

import tomllib


class SpinePathResolver:
    _SIBLING_NAME: str = "spine"
    _SIBREL_SRC: str = "src"
    _ENV_VAR: str = "EM_HSD_SPINE_PATH"
    _PYPROJECT_TOOL_SECTION: str = "em-hsd-v2"
    _PYPROJECT_KEY: str = "spine-path"

    @classmethod
    def resolve(cls) -> Path:
        """Return the resolved SPINE src directory.

        Resolution order:
          1. ``EM_HSD_SPINE_PATH`` environment variable.
          2. Sibling directory ``../spine/src`` relative to this file.
          3. ``[tool.em-hsd-v2] spine-path`` from ``pyproject.toml``.

        Raises:
            RuntimeError: if none of the above yield an existing directory.
        """
        env_path = os.environ.get(cls._ENV_VAR, "").strip()
        if env_path:
            candidate = Path(env_path).expanduser().resolve()
            if candidate.exists():
                return candidate

        sibling = (
            Path(__file__).resolve().parents[3].parent / cls._SIBLING_NAME / cls._SIBREL_SRC
        )
        if sibling.exists():
            return sibling

        pyproject_path = Path(__file__).resolve().parents[3] / "pyproject.toml"
        if pyproject_path.exists():
            from_pyproject = cls._from_pyproject(pyproject_path)
            if from_pyproject is not None and from_pyproject.exists():
                return from_pyproject

        raise RuntimeError(
            "Could not locate SPINE source directory. "
            f"Set the {cls._ENV_VAR} environment variable, "
            f"place the sibling directory '{cls._SIBLING_NAME}/{cls._SIBREL_SRC}' alongside the repo, "
            f"or configure [tool.{cls._PYPROJECT_TOOL_SECTION}] {cls._PYPROJECT_KEY} in pyproject.toml. "
            "See README for setup instructions."
        )

    @classmethod
    def _from_pyproject(cls, path: Path) -> Path | None:
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception:  # pragma: no cover
            return None

        try:
            raw = data["tool"][cls._PYPROJECT_TOOL_SECTION][cls._PYPROJECT_KEY]
        except KeyError:
            return None

        if not raw:
            return None

        candidate = Path(raw).expanduser().resolve()
        return candidate
