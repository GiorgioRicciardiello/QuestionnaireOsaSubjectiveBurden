"""Tests for exact (Clopper-Pearson) binomial intervals."""

import pytest

from questionnaire_osa.reporting.exact_intervals import clopper_pearson


def test_clopper_pearson_zero_successes_has_zero_lower_bound():
    lo, hi = clopper_pearson(0, 20)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 20), rel=1e-6)


def test_clopper_pearson_all_successes_has_unit_upper_bound():
    lo, hi = clopper_pearson(20, 20)
    assert hi == 1.0
    assert lo == pytest.approx(0.025 ** (1 / 20), rel=1e-6)


def test_clopper_pearson_known_value():
    # 7 of 10 successes: exact 95% CI is about (0.3475, 0.9333).
    lo, hi = clopper_pearson(7, 10)
    assert lo == pytest.approx(0.3475, abs=5e-4)
    assert hi == pytest.approx(0.9333, abs=5e-4)


def test_clopper_pearson_interval_contains_point_estimate():
    lo, hi = clopper_pearson(742, 1000)
    assert lo < 0.742 < hi
