"""Optional integration stub for the SPINE harness evaluate()."""

from __future__ import annotations

from typing import Any


def evaluate_with_harness(original_csv: str, privatized_csv: str, config_path: str) -> dict[str, Any]:
    """Call harness.evaluate if available; otherwise return a graceful degrade dict.

    This function does **not** import harness eagerly, so it is safe to import
    in environments where spine is not installed.
    """
    try:
        module = __import__("harness.evaluate", fromlist=["evaluate"])
        evaluate = module.evaluate
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "suggestion": "Install or symlink spine/src and ensure it is on PYTHONPATH.",
            "original_csv": original_csv,
            "privatized_csv": privatized_csv,
            "config_path": config_path,
        }

    try:
        result = evaluate(original_csv, privatized_csv)
        return {"available": True, "result": result}
    except Exception as exc:
        return {
            "available": True,
            "error": str(exc),
            "suggestion": "Harness present but evaluation failed; check CSV format and harness dependencies.",
        }
