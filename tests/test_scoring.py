from unlearnarena.scoring import normalized_recovery


def test_normalized_recovery_random_floor():
    assert normalized_recovery(0.25, 0.75) == 0.0


def test_normalized_recovery_original_ceiling():
    assert normalized_recovery(0.75, 0.75) == 1.0


def test_normalized_recovery_clips():
    assert normalized_recovery(0.9, 0.75) == 1.0
    assert normalized_recovery(0.1, 0.75) == 0.0
