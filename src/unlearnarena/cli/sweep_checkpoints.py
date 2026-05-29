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
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def infer_domain(name: str) -> str | None:
    lowered = name.lower()
    if "wmdp-bio" in lowered:
        return "bio"
    if "wmdp-cyber" in lowered:
        return "cyber"
    if "wmdp-chem" in lowered:
        return "chem"
    return None


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

        out_path = out_root / f"{checkpoint.name}.json"
        if args.skip_done and out_path.exists():
            summary[checkpoint.name] = json.loads(out_path.read_text())
            continue

        cmd = [
            sys.executable,
            "-m",
            "unlearnarena.cli.eval_checkpoint",
            "--model",
            str(checkpoint),
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
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])

        print(f"[run] {checkpoint.name} domain={domain}")
        subprocess.run(cmd, check=True)
        summary[checkpoint.name] = json.loads(out_path.read_text())

    summary_path = out_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
