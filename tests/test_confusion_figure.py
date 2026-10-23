"""Unit tests for the Figure 2 confusion-matrix aggregation and layout guards."""

import numpy as np
import pytest

from questionnaire_osa.plots.confusion_figure import (
    TARGETS, aggregate_counts, plot_confusion_figure, row_percent, sample_size,
)


def _folds(k: int, scale: int = 10) -> np.ndarray:
    base = np.arange(1, k * k + 1).reshape(k, k) * scale
    return np.stack([base + f for f in range(5)])


def test_mean_counts_are_rounded_fold_mean():
    folds = _folds(2)
    out = aggregate_counts(folds, "mean")
    assert out.dtype.kind == "i"
    assert np.array_equal(out, np.rint(folds.mean(axis=0)).astype(int))


def test_sum_counts_equal_total_across_folds():
    folds = _folds(4)
    assert aggregate_counts(folds, "sum").sum() == folds.sum()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        aggregate_counts(_folds(2), "median")


def test_row_percent_rows_sum_to_100_and_zero_row_is_safe():
    cm = np.array([[3, 1], [0, 0]])
    pct = row_percent(cm)
    assert pct[0].sum() == pytest.approx(100.0)
    assert np.all(pct[1] == 0.0)


def test_plot_requires_all_eight_targets(tmp_path):
    with pytest.raises(KeyError):
        plot_confusion_figure({"ahi_5": _folds(2)}, tmp_path / "f")


def test_targets_are_ordered_ahi_then_rdi():
    assert TARGETS[0][0] == "ahi_5" and TARGETS[1][-1] == "rdi_4class"
    assert all(len(row) == 4 for row in TARGETS)


def test_sample_size_uses_raw_totals_not_rounded_cells():
    folds = _folds(2, scale=1)
    assert sample_size(folds, "mean") == round(folds.sum() / 5)
    assert sample_size(folds, "sum") == folds.sum()
