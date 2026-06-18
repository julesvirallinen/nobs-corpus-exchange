from __future__ import annotations

import csv

from wrapper.csvio import CsvContractError
from wrapper.csvio import read_csv as read_privhsd_csv

CANONICAL_COLUMNS = ["ID", "Author", "Text", "HS"]
PRESERVED_COLUMNS = ["ID", "Author", "HS"]

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "ID": ("ID", "id"),
    "Author": ("Author", "author"),
    "Text": ("Text", "text"),
    "HS": ("HS", "hs"),
}
_EXTRA_KEY = "__extra__"

csv.field_size_limit(10 * 1024 * 1024)


def _resolve_columns(fieldnames: list[str]) -> dict[str, str]:    if not fieldnames:
        raise CsvContractError("empty file or missing header row.")
    lookup = {name: name for name in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                resolved[canonical] = alias
                break
        else:
            raise CsvContractError(
                f"missing required column for {canonical!r}; "
                f"expected one of {aliases}. Found: {', '.join(fieldnames)}"
            )
    return resolved


def read_csv_compat(path: str) -> tuple[list[str], list[dict[str, str]], dict[str, str]]:
    """Read CSV with ID/Author/Text/HS or id/author/text/hs headers.

    Returns (source_fieldnames, rows_with_canonical_keys, column_map).
    """
    try:
        fieldnames, rows = read_privhsd_csv(path)
        column_map = {c: c for c in CANONICAL_COLUMNS}
        return fieldnames, rows, column_map
    except CsvContractError:
        pass

    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, restkey=_EXTRA_KEY, restval=None)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise CsvContractError(f"{path!r}: empty file or missing header row.")
        column_map = _resolve_columns(list(fieldnames))
        rows_out: list[dict[str, str]] = []
        for i, raw in enumerate(reader):
            line = i + 2
            if _EXTRA_KEY in raw and raw[_EXTRA_KEY] not in (None, [], [""]):
                raise CsvContractError(
                    f"{path!r}: line {line}: too many fields for header {fieldnames}."
                )
            row: dict[str, str] = {}
            for canonical, source in column_map.items():
                if raw.get(source) is None:
                    raise CsvContractError(
                        f"{path!r}: line {line}: missing value for column {source!r}."
                    )
                row[canonical] = raw[source]
            rows_out.append(row)
        return list(fieldnames), rows_out, column_map


def write_csv_compat(
    path: str,
    source_fieldnames: list[str],
    rows: list[dict[str, str]],
    column_map: dict[str, str],
) -> None:    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=source_fieldnames, quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        for row in rows:
            out = {col: row.get(canonical, "") for canonical, col in column_map.items()}
            for col in source_fieldnames:
                out.setdefault(col, "")
            writer.writerow({k: out.get(k, "") for k in source_fieldnames})


def append_csv_row_compat(
    path: str,
    source_fieldnames: list[str],
    row: dict[str, str],
    column_map: dict[str, str],
    *,
    write_header: bool,
) -> None:    out = {col: row.get(canonical, "") for canonical, col in column_map.items()}
    mode = "w" if write_header else "a"
    with open(path, mode, encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=source_fieldnames, quoting=csv.QUOTE_MINIMAL,
        )
        if write_header:
            writer.writeheader()
        writer.writerow({k: out.get(k, "") for k in source_fieldnames})


def write_canonical_csv(path: str, rows: list[dict[str, str]]) -> None:    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CANONICAL_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in CANONICAL_COLUMNS})


def write_canonical_privatized_csv(path: str, rows: list[dict[str, str]]) -> None:    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["ID", "Text"], quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({"ID": row["ID"], "Text": row["Text"]})


class DiffCheckError(AssertionError):

def assert_preserved_compat(
    original_path: str,
    output_path: str,
    *,
    completed_rows: int | None = None,
) -> None:    _, original, _ = read_csv_compat(original_path)
    _, output, _ = read_csv_compat(output_path)
    if completed_rows is not None:
        original = original[:completed_rows]
        output = output[:completed_rows]
    problems: list[str] = []
    if len(original) != len(output):
        problems.append(
            f"row count differs: original={len(original)} output={len(output)}"
        )
    else:
        for i, (o, n) in enumerate(zip(original, output, strict=True)):
            for col in PRESERVED_COLUMNS:
                if o.get(col) != n.get(col):
                    problems.append(
                        f"row {i}: column {col!r} changed: "
                        f"{o.get(col)!r} -> {n.get(col)!r}"
                    )
    if problems:
        raise DiffCheckError(
            "Diff check FAILED — ID/Author/HS must be identical:\n  "
            + "\n  ".join(problems[:20])
        )
