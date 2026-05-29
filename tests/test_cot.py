from unlearnarena.answer_extraction import extract_mcq_answer


def test_extract_explicit_final_answer():
    assert extract_mcq_answer("After reasoning, final answer: C") == 2


def test_extract_bold_answer():
    assert extract_mcq_answer("The result is **B**.") == 1


def test_extract_fallback_last_letter():
    assert extract_mcq_answer("A is tempting. D") == 3
