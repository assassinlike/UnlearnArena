"""Dataset loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


WMDP_SUBSETS = {
    "bio": "wmdp-bio",
    "cyber": "wmdp-cyber",
    "chem": "wmdp-chem",
}


def load_wmdp_mcq(domain: str, data_dir: str | Path | None = None) -> list[dict[str, Any]]:
    """Load WMDP MCQ examples in a normalized format."""
    if domain not in WMDP_SUBSETS:
        raise ValueError(f"Unknown WMDP domain '{domain}'. Choose from {sorted(WMDP_SUBSETS)}.")

    if data_dir is not None:
        local = Path(data_dir) / "wmdp" / f"{domain}.json"
        if local.exists():
            data = json.loads(local.read_text())
            return [_normalize_mcq(row) for row in data]

    ds = load_dataset("cais/wmdp", WMDP_SUBSETS[domain], split="test")
    return [_normalize_mcq(row) for row in ds]


def load_texts(path: str | Path, limit: int | None = None) -> list[str]:
    """Load plain text examples from .txt, .json, or .jsonl."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        texts = [_row_to_text(row) for row in rows]
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {path}.")
        texts = [_row_to_text(row) for row in data]
    else:
        texts = [line.strip() for line in p.read_text().splitlines() if line.strip()]

    texts = [t for t in texts if t]
    return texts[:limit] if limit else texts


def mcq_to_texts(examples: list[dict[str, Any]]) -> list[str]:
    """Convert MCQ examples to simple QA text using the correct choice."""
    return [
        f"Question: {ex['question']}\nAnswer: {ex['choices'][ex['answer']]}"
        for ex in examples
    ]


def _normalize_mcq(row: Any) -> dict[str, Any]:
    return {
        "question": row["question"],
        "choices": list(row["choices"]),
        "answer": int(row["answer"]),
    }


def _row_to_text(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, dict):
        for key in ("text", "content", "abstract", "prompt"):
            val = row.get(key)
            if isinstance(val, str):
                return val
    raise ValueError(f"Cannot extract text from row: {row!r}")
