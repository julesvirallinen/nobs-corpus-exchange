"""CSV reading/writing that preserves every field except Text byte-for-byte.

The contract: input and output are CSV with columns ID, Author, Text, HS. Only
Text is replaced (one-to-one, same rows, same order, same count). ID, Author and
HS field *values* are preserved exactly, including commas, double quotes,
newlines and emoji inside any field, in UTF-8.

We use the stdlib ``csv`` module with ``newline=""`` and explicit UTF-8 so
embedded newlines/quotes round-trip correctly. Equality is checked on parsed
field *values* (the on-disk quoting style may differ but the values do not).
"""

from __future__ import annotations

import csv
from typing import Dict, List, Tuple

REQUIRED_COLUMNS = ["ID", "Author", "Text", "HS"]
_EXTRA_KEY = "__extra__"

# Allow large quoted Text fields (embedded newlines make rows long).
csv.field_size_limit(10 * 1024 * 1024)


class CsvContractError(ValueError):
    """Raised when the input does not satisfy the ID/Author/Text/HS contract."""


def read_csv(path: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """Read the CSV. Returns (fieldnames, rows). Raises CsvContractError on a
    missing header, missing required columns, or ragged (malformed) rows."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, restkey=_EXTRA_KEY, restval=None)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise CsvContractError(f"{path!r}: empty file or missing header row.")
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise CsvContractError(
                f"{path!r}: missing required column(s): {', '.join(missing)}. "
                f"Found: {', '.join(fieldnames)}"
            )
        rows: List[Dict[str, str]] = []
        for i, row in enumerate(reader):
            line = i + 2  # header is line 1
            if _EXTRA_KEY in row and row[_EXTRA_KEY] not in (None, [], [""]):
                raise CsvContractError(
                    f"{path!r}: line {line}: too many fields for header "
                    f"{fieldnames}."
                )
            for col in REQUIRED_COLUMNS:
                if row.get(col) is None:
                    raise CsvContractError(
                        f"{path!r}: line {line}: missing value for column {col!r} "
                        "(ragged row)."
                    )
            row.pop(_EXTRA_KEY, None)
            rows.append(row)
    return list(fieldnames), rows


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    """Write rows preserving column order. QUOTE_MINIMAL keeps the file tidy while
    still correctly quoting embedded commas/quotes/newlines."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
