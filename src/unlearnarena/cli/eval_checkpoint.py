"""Evaluate one checkpoint with UnlearnArena attacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unlearnarena.attacks import (
    answer_probe_attack,
    cot_generation_accuracy,
    cot_logit_accuracy,
    entropy_diagnostic,
    mcq_accuracy,
    mcq_accuracy_jailbreak_absolutely,
    mcq_accuracy_jailbreak_refuse,
    mcq_stats,
    mmlu_diagnostic,
    rebel_accuracy,
    relearning_attack,
)
from unlearnarena.data import load_texts, load_wmdp_mcq
from unlearnarena.models import load_causal_lm
from unlearnarena.scoring import normalized_recovery


ATTACKS = {
    "mcq",
    "jailbreak",
    "rebel",
    "cot_generation",
    "cot_logit",
    "probe",
    "relearn",
    "entropy",
    "mmlu",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint path or Hugging Face model id.")
    parser.add_argument("--base_model", default=None, help="Optional base model for original MCQ accuracy.")
    parser.add_argument("--domain", default="bio", choices=["bio", "cyber", "chem"])
    parser.add_argument("--attacks", default="mcq,jailbreak,rebel,cot_logit,entropy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--original_acc", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of MCQ examples.")

    parser.add_argument("--rebel_samples", type=int, default=200)
    parser.add_argument("--rebel_population", type=int, default=8)
    parser.add_argument("--rebel_generations", type=int, default=6)

    parser.add_argument("--cot_samples", type=int, default=None)
    parser.add_argument("--cot_max_tokens", type=int, default=400)
    parser.add_argument("--cot_fewshot_file", default=None)
    parser.add_argument("--include_cot_diagnostics", action="store_true")

    parser.add_argument("--probe_train_ratio", type=float, default=0.7)
    parser.add_argument("--probe_batch_size", type=int, default=8)
    parser.add_argument("--probe_save_dir", default=None)

    parser.add_argument("--same_domain_texts", default=None)
    parser.add_argument("--cross_domain_texts", default=None)
    parser.add_argument("--relearn_steps", type=int, default=200)
    parser.add_argument("--relearn_lr", type=float, default=2e-5)
    parser.add_argument("--relearn_n_texts", type=int, default=100)

    parser.add_argument("--original_mmlu_acc", type=float, default=None)
    parser.add_argument("--mmlu_n_per_subset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active = {item.strip() for item in args.attacks.split(",") if item.strip()}
    unknown = active - ATTACKS
    if unknown:
        raise SystemExit(f"Unknown attacks: {sorted(unknown)}. Choose from {sorted(ATTACKS)}.")

    examples = load_wmdp_mcq(args.domain, args.data_dir)
    if args.limit:
        examples = examples[: args.limit]

    model, tokenizer = load_causal_lm(args.model, args.device, args.dtype)

    original_acc = args.original_acc
    if original_acc is None:
        if args.base_model:
            base_model, base_tokenizer = load_causal_lm(args.base_model, args.device, args.dtype)
            original_acc = mcq_accuracy(base_model, base_tokenizer, examples, args.domain, args.device)
            del base_model, base_tokenizer
            torch.cuda.empty_cache()
        else:
            original_acc = mcq_accuracy(model, tokenizer, examples, args.domain, args.device)

    results = {
        "model": args.model,
        "domain": args.domain,
        "n_examples": len(examples),
        "original_acc": original_acc,
        "attacks": {},
    }

    if "mcq" in active:
        stats = mcq_stats(model, tokenizer, examples, args.domain, args.device)
        acc = stats["acc"]
        results["attacks"]["mcq"] = {
            "acc": acc,
            "score": normalized_recovery(acc, original_acc),
            "diagnostics": {
                "mean_normalized_entropy": stats["mean_normalized_entropy"],
                "mean_entropy_correct": stats["mean_entropy_correct"],
                "mean_entropy_wrong": stats["mean_entropy_wrong"],
                "entropy_asr": stats["entropy_asr"],
                "mislabeling_rate": stats["mislabeling_rate"],
                "n_confident_wrong": stats["n_confident_wrong"],
            },
        }

    if "jailbreak" in active:
        acc_refuse = mcq_accuracy_jailbreak_refuse(model, tokenizer, examples, args.domain, args.device)
        acc_abs = mcq_accuracy_jailbreak_absolutely(model, tokenizer, examples, args.domain, args.device)
        best = max(acc_refuse, acc_abs)
        results["attacks"]["jailbreak"] = {
            "acc_refuse": acc_refuse,
            "acc_absolutely": acc_abs,
            "best_acc": best,
            "score": normalized_recovery(best, original_acc),
        }

    if "rebel" in active:
        acc = rebel_accuracy(
            model,
            tokenizer,
            examples,
            args.domain,
            args.device,
            n_samples=args.rebel_samples,
            population_size=args.rebel_population,
            generations=args.rebel_generations,
        )
        results["attacks"]["rebel"] = {
            "acc": acc,
            "score": normalized_recovery(acc, original_acc),
        }

    cot_examples = examples[: args.cot_samples] if args.cot_samples else examples
    if "cot_generation" in active:
        acc = cot_generation_accuracy(
            model,
            tokenizer,
            cot_examples,
            args.domain,
            args.device,
            max_new_tokens=args.cot_max_tokens,
            include_diagnostics=args.include_cot_diagnostics,
        )
        if isinstance(acc, dict):
            cot_result = acc
            acc_value = cot_result["accuracy"]
        else:
            cot_result = None
            acc_value = acc
        results["attacks"]["cot_generation"] = {
            "acc": acc_value,
            "n": len(cot_examples),
            "score": normalized_recovery(acc_value, original_acc),
        }
        if cot_result and "diagnostics" in cot_result:
            results["attacks"]["cot_generation"]["diagnostics"] = cot_result["diagnostics"]

    if "cot_logit" in active:
        fewshot = Path(args.cot_fewshot_file).read_text() if args.cot_fewshot_file else ""
        acc = cot_logit_accuracy(
            model,
            tokenizer,
            cot_examples,
            args.domain,
            args.device,
            max_think_tokens=args.cot_max_tokens,
            fewshot=fewshot,
            include_diagnostics=args.include_cot_diagnostics,
        )
        if isinstance(acc, dict):
            cot_result = acc
            acc_value = cot_result["accuracy"]
        else:
            cot_result = None
            acc_value = acc
        results["attacks"]["cot_logit"] = {
            "acc": acc_value,
            "n": len(cot_examples),
            "score": normalized_recovery(acc_value, original_acc),
        }
        if cot_result and "diagnostics" in cot_result:
            results["attacks"]["cot_logit"]["diagnostics"] = cot_result["diagnostics"]

    if "probe" in active:
        results["attacks"]["probe"] = answer_probe_attack(
            model,
            tokenizer,
            examples,
            args.domain,
            original_acc,
            args.device,
            train_ratio=args.probe_train_ratio,
            batch_size=args.probe_batch_size,
            save_dir=args.probe_save_dir,
        )

    if "relearn" in active:
        if not args.same_domain_texts or not args.cross_domain_texts:
            raise SystemExit(
                "The relearn attack requires both --same_domain_texts and "
                "--cross_domain_texts. Same-domain texts should be domain-related "
                "but not the target forget/evaluation examples."
            )
        same_texts = load_texts(args.same_domain_texts, args.relearn_n_texts)
        cross_texts = load_texts(args.cross_domain_texts, args.relearn_n_texts)

        def eval_fn(m):
            return mcq_accuracy(m, tokenizer, examples[: min(100, len(examples))], args.domain, args.device)

        results["attacks"]["relearn"] = relearning_attack(
            model,
            tokenizer,
            same_texts,
            cross_texts,
            eval_fn,
            original_acc,
            args.device,
            steps=args.relearn_steps,
            lr=args.relearn_lr,
            n_finetune=args.relearn_n_texts,
        )

    if "entropy" in active:
        results["attacks"]["entropy"] = entropy_diagnostic(model, tokenizer, examples, args.device)

    if "mmlu" in active:
        results["attacks"]["mmlu"] = mmlu_diagnostic(
            model,
            tokenizer,
            original_mmlu_acc=args.original_mmlu_acc,
            device=args.device,
            data_dir=args.data_dir,
            n_per_subset=args.mmlu_n_per_subset,
        )

    output = json.dumps(results, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(output)
    print(output)


if __name__ == "__main__":
    main()
