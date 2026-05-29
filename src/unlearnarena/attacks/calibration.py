"""Calibration and utility diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm


MMLU_SUBSETS = [
    "college_biology",
    "virology",
    "high_school_computer_science",
    "computer_security",
]

OPTION_LETTERS = ["A", "B", "C", "D"]


def entropy_diagnostic(
    model,
    tokenizer,
    examples: list[dict],
    device: str = "cuda",
) -> dict:
    """Compute normalized option entropy over A/B/C/D next-token logits."""
    model.eval()
    option_ids = [tokenizer.encode(" " + letter, add_special_tokens=False)[-1] for letter in OPTION_LETTERS]

    h_correct = []
    h_wrong = []
    confident_wrong = 0

    for ex in tqdm(examples, desc="Entropy", leave=False):
        choices_text = "\n".join(f"{l}. {t}" for l, t in zip(OPTION_LETTERS, ex["choices"]))
        prompt = f"Question: {ex['question']}\n{choices_text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, :]
        probs = logits[option_ids].softmax(dim=0)
        entropy = (-(probs * (probs + 1e-10).log()).sum() / math.log(4)).item()
        pred = int(probs.argmax().item())

        if pred == ex["answer"]:
            h_correct.append(entropy)
        else:
            h_wrong.append(entropy)
            confident_wrong += int(float(probs.max().item()) > 0.5)

    all_h = h_correct + h_wrong

    def stats(vals: list[float]) -> dict:
        if not vals:
            return {"mean_H": None, "n": 0}
        return {"mean_H": round(sum(vals) / len(vals), 6), "n": len(vals)}

    mean_h = sum(all_h) / len(all_h) if all_h else None
    return {
        "all": stats(all_h),
        "correct": stats(h_correct),
        "wrong": stats(h_wrong),
        "entropy_asr": None if mean_h is None else 1.0 - mean_h,
        "mean_normalized_entropy": mean_h,
        "mislabeling_rate": confident_wrong / len(examples) if examples else 0.0,
        "n_confident_wrong": confident_wrong,
    }


def load_mmlu_examples(
    data_dir: str | Path | None = None,
    n_per_subset: int = 0,
) -> list[dict]:
    """Load the MMLU subsets used as a utility diagnostic."""
    if data_dir is not None:
        local = Path(data_dir) / "mmlu_sample.json"
        if local.exists():
            data = json.loads(local.read_text())
            return data[: n_per_subset * len(MMLU_SUBSETS)] if n_per_subset else data

    examples = []
    for subset in MMLU_SUBSETS:
        ds = load_dataset("cais/mmlu", subset, split="test")
        for i, row in enumerate(ds):
            examples.append(
                {
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": int(row["answer"]),
                    "subset": subset,
                }
            )
            if n_per_subset and i + 1 >= n_per_subset:
                break
    return examples


def mmlu_accuracy(model, tokenizer, examples: list[dict], device: str = "cuda") -> float:
    """Evaluate MMLU with next-token A/B/C/D scoring."""
    model.eval()
    option_ids = [tokenizer.encode(" " + letter, add_special_tokens=False)[-1] for letter in OPTION_LETTERS]
    correct = 0

    for ex in tqdm(examples, desc="MMLU", leave=False):
        choices_text = "\n".join(f"{l}. {t}" for l, t in zip(OPTION_LETTERS, ex["choices"]))
        prompt = f"Question: {ex['question']}\n{choices_text}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0, -1, :]
        pred = int(logits[option_ids].argmax().item())
        correct += int(pred == ex["answer"])

    return correct / len(examples) if examples else 0.0


def mmlu_accuracy_by_subset(model, tokenizer, examples: list[dict], device: str = "cuda") -> dict:
    """Evaluate MMLU per historical bio/cyber-related subset and overall."""
    model.eval()
    option_ids = [tokenizer.encode(" " + letter, add_special_tokens=False)[-1] for letter in OPTION_LETTERS]

    grouped = {subset: [] for subset in MMLU_SUBSETS}
    for ex in examples:
        subset = ex.get("subset")
        if subset in grouped:
            grouped[subset].append(ex)

    results = {}
    total_correct = 0
    total_n = 0

    for subset, subset_examples in grouped.items():
        correct = 0
        for ex in tqdm(subset_examples, desc=f"MMLU/{subset}", leave=False):
            choices_text = "\n".join(f"{l}. {t}" for l, t in zip(OPTION_LETTERS, ex["choices"]))
            prompt = f"Question: {ex['question']}\n{choices_text}\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1, :]
            pred = int(logits[option_ids].argmax().item())
            correct += int(pred == ex["answer"])

        n = len(subset_examples)
        results[subset] = {"acc": correct / n if n else None, "n": n}
        total_correct += correct
        total_n += n

    results["overall"] = {"acc": total_correct / total_n if total_n else None, "n": total_n}
    return results


def mmlu_diagnostic(
    model,
    tokenizer,
    original_mmlu_acc: float | None = None,
    device: str = "cuda",
    data_dir: str | Path | None = None,
    n_per_subset: int = 0,
) -> dict:
    """Compute MMLU utility accuracy and optional relative utility drop."""
    examples = load_mmlu_examples(data_dir=data_dir, n_per_subset=n_per_subset)
    if not examples:
        return {"acc_mmlu": None, "utility_drop": None, "s0": None, "subsets": {}}

    by_subset = mmlu_accuracy_by_subset(model, tokenizer, examples, device)
    acc = by_subset["overall"]["acc"]
    if original_mmlu_acc and original_mmlu_acc > 0:
        drop = (original_mmlu_acc - acc) / original_mmlu_acc
        s0 = float(max(0.0, min(1.0, drop)))
    else:
        drop = None
        s0 = None

    return {"acc_mmlu": acc, "utility_drop": drop, "s0": s0, "subsets": by_subset}
