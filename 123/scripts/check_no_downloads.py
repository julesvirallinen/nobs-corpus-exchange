#!/usr/bin/env python3
"""Static check that model downloads are gated, not direct."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from em_hsd.core.policy import DownloadPolicy

ROOT = Path(__file__).resolve().parents[1] / "src" / "em_hsd"

_DOWNLOAD_CALLS = {
    "transformers.AutoModelForCausalLM.from_pretrained",
    "transformers.AutoModelForSequenceClassification.from_pretrained",
    "transformers.AutoModelForMaskedLM.from_pretrained",
    "transformers.AutoTokenizer.from_pretrained",
    "unsloth.FastLanguageModel.from_pretrained",
    "sentence_transformers.SentenceTransformer",
    "torch.hub.download_url_to_file",
}


def _qual_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def scan(path: Path) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for py in path.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            print(f"WARN: {py}: syntax error: {exc}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _qual_name(node.func)
                if name in _DOWNLOAD_CALLS:
                    hits.append((str(py.relative_to(path.parent.parent)), node.lineno or 0, name))
    return hits


def main() -> int:
    print(DownloadPolicy.allowed_message())
    hits = scan(ROOT)
    if not hits:
        print("OK: no direct download calls found in src/em_hsd/.")
        return 0

    print("Direct download calls detected (these must be gated by ResourceManager):")
    for rel, lineno, name in hits:
        print(f"  {rel}:{lineno}: {name}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
