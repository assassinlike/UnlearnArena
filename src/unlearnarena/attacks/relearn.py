"""Fine-tuning based relearning attacks."""

from __future__ import annotations

import torch
from tqdm import tqdm

from unlearnarena.scoring import RANDOM_MCQ_ACC, normalized_recovery


def finetune_and_measure(
    model,
    tokenizer,
    finetune_texts: list[str],
    eval_fn,
    steps: int,
    lr: float,
    batch_size: int,
    max_length: int,
    device: str,
) -> float:
    """Fine-tune temporarily and restore original weights after measuring recovery."""
    if not finetune_texts:
        return eval_fn(model)

    cpu_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    enc = tokenizer(
        finetune_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr * 10, momentum=0.9)

    try:
        for _ in tqdm(range(steps), desc="relearn-finetune", leave=False):
            idx = torch.randperm(input_ids.shape[0], device=input_ids.device)[:batch_size]
            batch_ids = input_ids[idx]
            batch_mask = attention_mask[idx]
            optimizer.zero_grad()
            loss = model(input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids).loss
            loss.backward()
            optimizer.step()

        model.eval()
        acc = eval_fn(model)
    finally:
        model.load_state_dict({k: v.to(device) for k, v in cpu_state.items()})
        model.eval()

    return acc


def relearning_attack(
    model,
    tokenizer,
    same_domain_texts: list[str],
    cross_domain_texts: list[str],
    eval_fn,
    original_acc: float,
    device: str = "cuda",
    steps: int = 200,
    lr: float = 2e-5,
    n_finetune: int = 100,
    batch_size: int = 4,
    max_length: int = 256,
) -> dict:
    """
    Relearning attack with same-domain and cross-domain fine-tuning.

    The historical minimal-data variant is intentionally omitted. Same-domain
    follows the newer implementation and is compatible with the older same-domain
    setting when the caller provides non-forget, domain-related texts.
    """
    acc_same = finetune_and_measure(
        model,
        tokenizer,
        same_domain_texts[:n_finetune],
        eval_fn,
        steps,
        lr,
        batch_size,
        max_length,
        device,
    )
    acc_cross = finetune_and_measure(
        model,
        tokenizer,
        cross_domain_texts[:n_finetune],
        eval_fn,
        steps,
        lr,
        batch_size,
        max_length,
        device,
    )

    best_acc = max(acc_same, acc_cross)
    return {
        "score": normalized_recovery(best_acc, original_acc, RANDOM_MCQ_ACC),
        "acc_same_domain": acc_same,
        "acc_cross_domain": acc_cross,
        "best_acc": best_acc,
    }
