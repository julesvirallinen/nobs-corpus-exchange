"""The diff check: ID/Author/HS preserved, Text replaced one-to-one.

Runs both as a pytest and as the wrapper's built-in post-write check (which
fails loudly, non-zero exit, if violated).
"""

from __future__ import annotations

from typing import Dict, List

from .csvio import REQUIRED_COLUMNS, read_csv

PRESERVED_COLUMNS = ["ID", "Author", "HS"]


class DiffCheckError(AssertionError):
    """Raised when the output violates the preservation contract."""


def diff_field_values(
    original: List[Dict[str, str]],
    output: List[Dict[str, str]],
    preserved=PRESERVED_COLUMNS,
) -> List[str]:
    """Return a list of human-readable violations (empty == OK)."""
    problems: List[str] = []
    if len(original) != len(output):
        problems.append(
            f"row count differs: original={len(original)} output={len(output)}"
        )
        return problems  # row-by-row comparison is meaningless past this point
    for i, (o, n) in enumerate(zip(original, output)):
        for col in preserved:
            if o.get(col) != n.get(col):
                problems.append(
                    f"row {i}: column {col!r} changed: "
                    f"{o.get(col)!r} -> {n.get(col)!r}"
                )
    return problems


def check_files(original_path: str, output_path: str) -> List[str]:
    """Re-read both CSVs from disk and diff the preserved columns."""
    _, original = read_csv(original_path)
    _, output = read_csv(output_path)
    return diff_field_values(original, output)


def assert_preserved(original_path: str, output_path: str) -> None:
    problems = check_files(original_path, output_path)
    if problems:
        raise DiffCheckError(
            "Diff check FAILED — ID/Author/HS must be byte-identical and rows "
            "one-to-one:\n  " + "\n  ".join(problems[:20])
        )
