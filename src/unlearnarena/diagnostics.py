"""Reusable diagnostics for attack outputs."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean


def ngram_entropy(items: list, n: int) -> float:
    """Return negative n-gram entropy sum for a sequence of hashable items."""
    if len(items) < n:
        return 0.0
    ngrams = [tuple(items[i : i + n]) for i in range(len(items) - n + 1)]
    total = len(ngrams)
    counts = Counter(ngrams)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def char_generation_entropy(text: str) -> float:
    """Historical character-level bigram/trigram generation entropy."""
    chars = list(text)
    return (2 / 3) * ngram_entropy(chars, 2) + (4 / 3) * ngram_entropy(chars, 3)


def word_generation_entropy(text: str) -> float:
    """Word-level bigram/trigram generation entropy."""
    words = text.split()
    return (2 / 3) * ngram_entropy(words, 2) + (4 / 3) * ngram_entropy(words, 3)


def is_repetitive(text: str, threshold: float = 0.5) -> bool:
    """Detect strong character or bigram repetition in generated text."""
    if len(text) < 10:
        return False
    if any(text.count(ch) / len(text) > threshold for ch in set(text)):
        return True
    if len(text) > 20:
        bigrams = [text[i : i + 2] for i in range(len(text) - 1)]
        if any(bigrams.count(bg) / len(bigrams) > threshold for bg in set(bigrams)):
            return True
    return False


def follows_cot_instruction(text: str, stop_phrase: str) -> bool:
    """Whether generated reasoning includes the required CoT stop/instruction phrase."""
    return stop_phrase in text


def generation_diagnostics(text: str, stop_phrase: str | None = None) -> dict:
    """Per-generation diagnostics for CoT outputs."""
    has_stop = bool(stop_phrase and stop_phrase in text)
    return {
        "char_entropy": char_generation_entropy(text),
        "word_entropy": word_generation_entropy(text),
        "repetitive": is_repetitive(text),
        "has_stop_phrase": has_stop,
        "instruction_followed": has_stop if stop_phrase else None,
    }


def summarize_generation_diagnostics(rows: list[dict]) -> dict:
    """Aggregate per-generation diagnostics."""
    if not rows:
        return {
            "mean_char_entropy": None,
            "mean_word_entropy": None,
            "repetition_rate": None,
            "stop_rate": None,
            "instruction_follow_rate": None,
        }

    def values(key: str) -> list:
        return [row[key] for row in rows if row.get(key) is not None]

    stop_vals = values("has_stop_phrase")
    instruction_vals = values("instruction_followed")
    return {
        "mean_char_entropy": mean(values("char_entropy")),
        "mean_word_entropy": mean(values("word_entropy")),
        "repetition_rate": mean(values("repetitive")),
        "stop_rate": mean(stop_vals) if stop_vals else None,
        "instruction_follow_rate": mean(instruction_vals) if instruction_vals else None,
    }
