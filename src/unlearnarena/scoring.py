"""Shared scoring helpers."""

from __future__ import annotations

import numpy as np


RANDOM_MCQ_ACC = 0.25


def normalized_recovery(
    recovered_acc: float,
    original_acc: float,
    random_acc: float = RANDOM_MCQ_ACC,
) -> float:
    """Normalize an attack accuracy to [0, 1] relative to random and original accuracy."""
    denom = original_acc - random_acc
    if abs(denom) < 1e-8:
        return 0.0
    return float(np.clip((recovered_acc - random_acc) / denom, 0.0, 1.0))


def lp_mean(scores: list[float], p: int = 4) -> float:
    """Lp-style aggregate used by the historical experiments."""
    if not scores:
        return 0.0
    return float((sum(s**p for s in scores) ** (1.0 / p)) / len(scores))
