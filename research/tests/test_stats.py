"""CPU tests for analysis/stats.py's exact sign-flip test (docs/prereg_v3.md, Statistics)."""

import pytest

from research.analysis.stats import sign_flip_p


def test_smallest_p_is_one_over_two_to_the_n():
    assert sign_flip_p([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]) == 1 / 128  # seven seeds can pass alpha = 0.025
    assert sign_flip_p([0.1, 0.2, 0.3, 0.4, 0.5]) == 1 / 32  # five cannot (prereg-v3 open decision 1)


def test_hand_counted_mixed_case():
    # sums of (+-1, +-2, +-0.5): 3.5 and 2.5 reach the observed 2.5, the other six do not
    assert sign_flip_p([1.0, 2.0, -0.5]) == 2 / 8


def test_ties_count_and_no_effect_gives_one():
    assert sign_flip_p([0.0, 0.0, 0.0]) == 1.0
    assert sign_flip_p([0.1, 0.1]) == 1 / 4
    assert sign_flip_p([0.1, 0.2, 0.3]) < sign_flip_p([0.1, -0.2, 0.3]) <= 1.0
    assert sign_flip_p([-0.1, -0.2, -0.3]) == 1.0  # one-sided: a negative effect is no evidence


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], [1.0, float("nan")], [0.1] * 17])
def test_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        sign_flip_p(bad)
