from __future__ import annotations

from typing import Dict, List

from .csvio import REQUIRED_COLUMNS, read_csv

PRESERVED_COLUMNS = ["ID", "Author", "HS"]


class DiffCheckError(AssertionError):

def diff_field_values(
    original: List[Dict[str, str]],
    output: List[Dict[str, str]],
    preserved=PRESERVED_COLUMNS,
) -> List[str]:    problems: List[str] = []
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


def check_files(original_path: str, output_path: str) -> List[str]:    _, original = read_csv(original_path)
    _, output = read_csv(output_path)
    return diff_field_values(original, output)


def assert_preserved(original_path: str, output_path: str) -> None:
    problems = check_files(original_path, output_path)
    if problems:
        raise DiffCheckError(
            "Diff check FAILED — ID/Author/HS must be byte-identical and rows "
            "one-to-one:\n  " + "\n  ".join(problems[:20])
        )
