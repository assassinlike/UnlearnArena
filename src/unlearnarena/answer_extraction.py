"""Lightweight answer extraction helpers."""

from __future__ import annotations

import re


OPTION_LETTERS = ["A", "B", "C", "D"]


def extract_mcq_answer(text: str) -> int:
    """Extract A/B/C/D from generated text. Returns 0-3, or -1 if no answer is found."""
    patterns = [
        r"(?:final answer|the answer is|answer is|therefore)[:\s]*\**([ABCD])\**",
        r"\*\*([ABCD])\*\*",
        r"^([ABCD])[)\.]",
        r"\b([ABCD])\b(?:\s*(?:is correct|is the answer))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return OPTION_LETTERS.index(match.group(1).upper())

    matches = re.findall(r"\b([ABCD])\b", text.upper())
    if matches:
        return OPTION_LETTERS.index(matches[-1])
    return -1
