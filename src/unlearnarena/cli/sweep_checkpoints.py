"""Evaluate a directory of checkpoints with UnlearnArena."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Directory containing checkpoints.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--domain", default=None, choices=["bio", "cyber", "chem"])
    parser.add_argument("--attacks", default="mcq,jailbreak,rebel,cot_logit,entropy")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--only", default=None, help="Substring filter on checkpoint directory names.")
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--original_acc", type=float, default=None)
    parser.add_argument("--original_mmlu_acc", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rebel_samples", type=int, default=None)
    parser.add_argument("--cot_samples", type=int, default=None)
    parser.add_argument("--cot_max_tokens", type=int, default=None)
    parser.add_argument("--mmlu_n_per_subset", type=int, default=None)
    parser.add_argument("--include_cot_diagnostics", action="store_true")
    return parser.parse_args()


def infer_domain(name: str) -> str | None:
    lowered = name.lower()
    if "wmdp-bio" in lowered or "bio" in lowered:
        return "bio"
    if "wmdp-cyber" in lowered or "cyber" in lowered:
        return "cyber"
    if "wmdp-chem" in lowered:
        return "chem"
    return None


def resolve_checkpoint_path(path: Path) -> Path:
    """
    Resolve either a direct HF checkpoint directory or a local HF cache wrapper.

    The shared WMDP cache is organized as:
        repo_alias/models--owner--repo/snapshots/<revision>/config.json
    while transformers expects the snapshot directory itself.
    """
    if (path / "config.json").exists():
        return path

    configs = sorted(path.glob("models--*/snapshots/*/config.json"))
    if not configs:
        raise FileNotFoundError(f"Could not find config.json under {path}")
    return configs[-1].parent


def main() -> None:
    args = parse_args()
    model_root = Path(args.model_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(p for p in model_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if args.only:
        checkpoints = [p for p in checkpoints if args.only.lower() in p.name.lower()]

    summary = {}
    for checkpoint in checkpoints:
        domain = args.domain or infer_domain(checkpoint.name)
        if domain is None:
            print(f"[skip] Cannot infer domain for {checkpoint.name}; pass --domain.", file=sys.stderr)
            continue

        model_path = resolve_checkpoint_path(checkpoint)
        out_path = out_root / f"{checkpoint.name}.json"
        if args.skip_done and out_path.exists():
            summary[checkpoint.name] = json.loads(out_path.read_text())
            continue

        cmd = [
            sys.executable,
            "-m",
            "unlearnarena.cli.eval_checkpoint",
            "--model",
            str(model_path),
            "--domain",
            domain,
            "--attacks",
            args.attacks,
            "--device",
            args.device,
            "--dtype",
            args.dtype,
            "--out",
            str(out_path),
        ]
        if args.data_dir:
            cmd.extend(["--data_dir", args.data_dir])
        if args.base_model:
            cmd.extend(["--base_model", args.base_model])
        if args.original_acc is not None:
            cmd.extend(["--original_acc", str(args.original_acc)])
        if args.original_mmlu_acc is not None:
            cmd.extend(["--original_mmlu_acc", str(args.original_mmlu_acc)])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.rebel_samples is not None:
            cmd.extend(["--rebel_samples", str(args.rebel_samples)])
        if args.cot_samples is not None:
            cmd.extend(["--cot_samples", str(args.cot_samples)])
        if args.cot_max_tokens is not None:
            cmd.extend(["--cot_max_tokens", str(args.cot_max_tokens)])
        if args.mmlu_n_per_subset is not None:
            cmd.extend(["--mmlu_n_per_subset", str(args.mmlu_n_per_subset)])
        if args.include_cot_diagnostics:
            cmd.append("--include_cot_diagnostics")

        print(f"[run] {checkpoint.name} domain={domain} model_path={model_path}")
        subprocess.run(cmd, check=True)
        summary[checkpoint.name] = json.loads(out_path.read_text())

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
