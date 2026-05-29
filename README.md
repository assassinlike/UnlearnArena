# UnlearnArena

UnlearnArena is a clean, attack-focused evaluation toolkit for auditing language
models that have been unlearned on WMDP-style knowledge. It keeps only attack
and diagnostic code: it does not train or apply unlearning methods.

The initial implementation is distilled from historical ULARS experiments. When
multiple versions of the same attack existed, the later implementation was used
as the base unless noted below.

## Attacks

- `mcq`: direct WMDP multiple-choice next-token scoring.
- `jailbreak`: two direct MCQ prompt variants (`refuse`, `absolutely`).
- `rebel`: REBEL-style natural-language evolutionary prompt search.
- `cot_generation`: generates a chain-of-thought answer and extracts `A/B/C/D`
  from the generated text.
- `cot_logit`: generates reasoning, appends it to the prompt, then scores
  `A/B/C/D` with next-token logits.
- `probe`: trains a linear probe on hidden states to predict the correct MCQ
  option, using a train/test split of the target WMDP examples.
- `relearn`: temporarily fine-tunes the checkpoint on same-domain and
  cross-domain texts, then measures WMDP accuracy recovery.
- `entropy`: computes normalized option entropy and confident-wrong diagnostics.
- `mmlu`: evaluates utility on the historical bio/cyber-adjacent MMLU subsets:
  `college_biology`, `virology`, `high_school_computer_science`, and
  `computer_security`.

The probe attack intentionally uses hidden states to predict the correct option.
A future extension can add transfer probing where the probe is trained on benign
multiple-choice questions and tested on unlearned WMDP content.

## Install

```bash
cd /zfspool/zangyihe/home/UnlearnArena
pip install -e .
```

## Evaluate One Checkpoint

```bash
unlearnarena-eval \
  --model /path/to/unlearned-checkpoint \
  --base_model /path/to/base-model \
  --domain bio \
  --attacks mcq,jailbreak,rebel,cot_logit,entropy,mmlu \
  --device cuda:0 \
  --out outputs/model_bio.json
```

If `--base_model` or `--original_acc` is not provided, the checkpoint itself is
used as the original-accuracy baseline. For unlearned checkpoints, prefer
passing either the true base model or a precomputed `--original_acc`.

To add CoT generation diagnostics, pass:

```bash
unlearnarena-eval \
  --model /path/to/unlearned-checkpoint \
  --domain bio \
  --attacks cot_generation,cot_logit \
  --include_cot_diagnostics
```

This adds char-level and word-level generation entropy, repetition rate,
stop-phrase rate, and instruction-following rate to the CoT attack outputs.
The `mcq` attack always includes option entropy diagnostics.

## Sweep A Checkpoint Directory

```bash
unlearnarena-sweep \
  --model_dir /path/to/wmdp-models \
  --base_model /path/to/base-model \
  --out_dir outputs/wmdp_sweep \
  --attacks mcq,jailbreak,rebel,cot_logit,entropy \
  --device cuda:0
```

The sweep command infers `bio`, `cyber`, or `chem` from checkpoint directory
names containing `wmdp-bio`, `wmdp-cyber`, or `wmdp-chem`. Use `--domain` to
override.

## Relearning Inputs

The relearning attack accepts:

```bash
unlearnarena-eval \
  --model /path/to/unlearned-checkpoint \
  --domain bio \
  --attacks relearn \
  --same_domain_texts /path/to/domain_texts.jsonl \
  --cross_domain_texts /path/to/cross_domain_texts.jsonl
```

The relearning attack requires both `--same_domain_texts` and
`--cross_domain_texts`. Provide non-forget domain-related texts for same-domain
relearning and unrelated benign texts for cross-domain relearning.

## Notes

- All current attacks are designed around four-choice MCQ evaluation.
- This repository does not include hazardous few-shot examples. The
  `cot_logit` attack accepts `--cot_fewshot_file` if an experiment needs a
  controlled few-shot prompt.
- The MMLU diagnostic reports per-subset and overall accuracy. Use
  `--mmlu_n_per_subset` for a quick sample and `--original_mmlu_acc` to compute
  utility drop.
- Absolute paths from the historical experiments have been removed from the
  package API and replaced with CLI arguments.
