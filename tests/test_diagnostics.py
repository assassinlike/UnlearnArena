from unlearnarena.diagnostics import (
    char_generation_entropy,
    generation_diagnostics,
    is_repetitive,
    summarize_generation_diagnostics,
    word_generation_entropy,
)


def test_generation_entropy_positive_for_varied_text():
    text = "alpha beta gamma delta"
    assert char_generation_entropy(text) > 0
    assert word_generation_entropy(text) > 0


def test_repetition_detection():
    assert is_repetitive("aaaaaaaaaaaaaaaaaaaa")
    assert not is_repetitive("alpha beta gamma delta")


def test_generation_diagnostics_summary():
    stop = "Having thought through this step by step:"
    rows = [
        generation_diagnostics(f"reasoning {stop}", stop),
        generation_diagnostics("aaaaaaaaaaaaaaaaaaaa", stop),
    ]
    summary = summarize_generation_diagnostics(rows)
    assert summary["stop_rate"] == 0.5
    assert summary["instruction_follow_rate"] == 0.5
    assert summary["repetition_rate"] == 0.5
