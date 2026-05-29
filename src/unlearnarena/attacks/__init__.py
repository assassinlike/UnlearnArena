"""Attack implementations with lazy exports."""

from __future__ import annotations

_EXPORTS = {
    "answer_probe_attack": ("unlearnarena.attacks.probe", "answer_probe_attack"),
    "cot_generation_accuracy": ("unlearnarena.attacks.cot", "cot_generation_accuracy"),
    "cot_logit_accuracy": ("unlearnarena.attacks.cot", "cot_logit_accuracy"),
    "entropy_diagnostic": ("unlearnarena.attacks.calibration", "entropy_diagnostic"),
    "mcq_accuracy": ("unlearnarena.attacks.mcq", "mcq_accuracy"),
    "mcq_stats": ("unlearnarena.attacks.mcq", "mcq_stats"),
    "mcq_accuracy_jailbreak_absolutely": (
        "unlearnarena.attacks.mcq",
        "mcq_accuracy_jailbreak_absolutely",
    ),
    "mcq_accuracy_jailbreak_refuse": ("unlearnarena.attacks.mcq", "mcq_accuracy_jailbreak_refuse"),
    "mmlu_diagnostic": ("unlearnarena.attacks.calibration", "mmlu_diagnostic"),
    "rebel_accuracy": ("unlearnarena.attacks.rebel", "rebel_accuracy"),
    "relearning_attack": ("unlearnarena.attacks.relearn", "relearning_attack"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attr = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr)
    globals()[name] = value
    return value
