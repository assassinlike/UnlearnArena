"""Model loading helpers."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(
    model_name_or_path: str,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    trust_remote_code: bool = True,
):
    """Load a causal LM and tokenizer for attack evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        device_map=device,
    )
    model.eval()
    return model, tokenizer
