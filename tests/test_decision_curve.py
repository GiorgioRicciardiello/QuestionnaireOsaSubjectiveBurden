"""Unit tests for decision-curve net benefit (Vickers & Elkin, 2006)."""

import numpy as np
import pytest

from questionnaire_osa.evaluation.decision_curve import (
    decision_curve_frame,
    delta_net_benefit_bootstrap,
    net_benefit,
    net_benefit_curve,
    treat_all_net_benefit,
)


def test_net_benefit_hand_computed_value():
    # 4 cases, 6 controls; threshold 0.2 flags 3 cases and 2 controls.
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.3, 0.1, 0.5, 0.25, 0.05, 0.05, 0.05, 0.05])
    # TP = 3, FP = 2, n = 10, odds = 0.25
    expected = 3 / 10 - 2 / 10 * 0.25
    assert net_benefit(y, p, 0.2) == pytest.approx(expected)


def test_net_benefit_perfect_model_equals_prevalence_at_low_threshold():
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0, 0, 0])
    p = y.astype(float)
    assert net_benefit(y, p, 0.5) == pytest.approx(y.mean())


def test_treat_all_matches_model_that_flags_everyone():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 1])
    everyone = np.ones_like(y, dtype=float)
    for pt in (0.05, 0.2, 0.5):
        assert treat_all_net_benefit(y, pt) == pytest.approx(
            net_benefit(y, everyone, pt)
        )


def test_net_benefit_rejects_threshold_outside_open_unit_interval():
    y = np.array([0, 1])
    p = np.array([0.1, 0.9])
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            net_benefit(y, p, bad)


def test_net_benefit_curve_default_grid_is_0p01_to_0p80():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.9])
    curve = net_benefit_curve(y, p)
    assert curve["threshold"].iloc[0] == pytest.approx(0.01)
    assert curve["threshold"].iloc[-1] == pytest.approx(0.80)
    assert len(curve) == 80
    assert list(curve.columns) == ["threshold", "net_benefit"]


def test_decision_curve_frame_has_model_and_reference_strategies():
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.8, 0.4, 0.7, 0.6, 0.2])
    thresholds = np.array([0.1, 0.5])
    frame = decision_curve_frame(y, {"model": p}, thresholds=thresholds)
    assert set(frame["strategy"]) == {"model", "treat_all", "treat_none"}
    none = frame[frame["strategy"] == "treat_none"]
    assert (none["net_benefit"] == 0.0).all()


def test_delta_bootstrap_point_estimate_matches_direct_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good = np.clip(y * 0.4 + rng.normal(0.3, 0.2, y.size), 0, 1)
    weak = np.clip(y * 0.1 + rng.normal(0.3, 0.2, y.size), 0, 1)
    out = delta_net_benefit_bootstrap(
        y, good, weak, thresholds=[0.2], n_resamples=200, random_state=1,
    )
    row = out.iloc[0]
    assert row["delta_nb"] == pytest.approx(
        net_benefit(y, good, 0.2) - net_benefit(y, weak, 0.2)
    )
    assert np.isfinite(row["ci_lo"]) and np.isfinite(row["ci_hi"])
    assert row["ci_lo"] < row["ci_hi"]


def test_delta_bootstrap_is_reproducible_with_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    a = np.clip(y * 0.3 + rng.normal(0.3, 0.2, y.size), 0, 1)
    b = np.clip(y * 0.2 + rng.normal(0.3, 0.2, y.size), 0, 1)
    first = delta_net_benefit_bootstrap(y, a, b, thresholds=[0.1, 0.3], n_resamples=100, random_state=7)
    second = delta_net_benefit_bootstrap(y, a, b, thresholds=[0.1, 0.3], n_resamples=100, random_state=7)
    assert first.equals(second)


def test_delta_bootstrap_stratifies_by_outcome():
    # Each resample must keep the case count fixed, so with all cases
    # identical in probability the CI collapses to zero width.
    y = np.array([1] * 20 + [0] * 80)
    a = np.where(y == 1, 0.9, 0.1).astype(float)
    b = a.copy()
    out = delta_net_benefit_bootstrap(y, a, b, thresholds=[0.2], n_resamples=50, random_state=0)
    assert out.iloc[0]["ci_lo"] == pytest.approx(0.0)
    assert out.iloc[0]["ci_hi"] == pytest.approx(0.0)


def test_calibration_summary_perfect_probabilities_have_unit_slope():
    from questionnaire_osa.evaluation.decision_curve import calibration_summary

    rng = np.random.default_rng(5)
    p = rng.uniform(0.05, 0.95, 5000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    row = calibration_summary(y, p)
    assert row["calibration_slope"] == pytest.approx(1.0, abs=0.1)
    assert row["calibration_intercept"] == pytest.approx(0.0, abs=0.1)
    assert row["mean_predicted"] == pytest.approx(p.mean())
