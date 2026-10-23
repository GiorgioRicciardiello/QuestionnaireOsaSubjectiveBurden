"""Unit tests for the DeLong / Wilson statistics behind Table 2, S9 and S11."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from questionnaire_osa.reporting.reference_run import (
    build_table2, delong, endpoint_metrics, wilson,
)


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    strong = y * 1.0 + rng.normal(0, 1.0, y.size)
    weak = y * 0.5 + rng.normal(0, 1.0, y.size)
    return y, strong, weak


def test_delong_auc_matches_sklearn(synthetic):
    y, strong, weak = synthetic
    aucs, _ = delong(y, [strong, weak])
    assert aucs[0] == pytest.approx(roc_auc_score(y, strong), abs=1e-12)
    assert aucs[1] == pytest.approx(roc_auc_score(y, weak), abs=1e-12)


def test_delong_handles_ties():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.5, 0.5, 0.9, 0.5, 0.5])
    aucs, _ = delong(y, [s])
    assert aucs[0] == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_delong_variance_matches_bootstrap(synthetic):
    y, strong, weak = synthetic
    _, cov = delong(y, [strong, weak])
    rng = np.random.default_rng(1)
    diffs = []
    for _ in range(400):
        i = rng.integers(0, y.size, y.size)
        diffs.append(roc_auc_score(y[i], strong[i]) - roc_auc_score(y[i], weak[i]))
    se_delong = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert se_delong == pytest.approx(np.std(diffs), rel=0.15)


def test_wilson_known_value():
    # 8/10: Wilson 95% interval (0.490, 0.943)
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert np.isnan(wilson(0, 0)[0])


def _feature_sets(y, scores):
    fs = {}
    for name, s in scores.items():
        fs[name] = {"all_predictions": {"t": {
            "y_true": y, "y_proba": s, "y_pred": (s >= np.median(s)).astype(int)}}}
    return fs


def test_endpoint_metrics_consistent(synthetic):
    y, strong, weak = synthetic
    fs = _feature_sets(y, {"all_features": strong, "dem_only": weak, "no_dem": weak * 0.5})
    r = endpoint_metrics(fs, "t")
    m = r["fs"]["all_features"]
    assert m["auroc"] == pytest.approx(m["auroc_sklearn"], abs=1e-12)
    assert m["f1"] == pytest.approx(2 * m["precision"] * m["sensitivity"]
                                    / (m["precision"] + m["sensitivity"]))
    lo, hi = r["diffs"][("all_features", "dem_only")]["ci"]
    assert lo < r["diffs"][("all_features", "dem_only")]["delta"] < hi


def test_endpoint_metrics_rejects_mismatched_patients(synthetic):
    y, strong, weak = synthetic
    fs = _feature_sets(y, {"all_features": strong, "dem_only": weak, "no_dem": weak})
    fs["no_dem"]["all_predictions"]["t"]["y_true"] = 1 - y
    with pytest.raises(ValueError, match="same test patients"):
        endpoint_metrics(fs, "t")


def test_build_table2_layout(synthetic):
    y, strong, weak = synthetic
    fs = _feature_sets(y, {"all_features": strong, "dem_only": weak, "no_dem": weak})
    results = {"t": endpoint_metrics(fs, "t")}
    df = build_table2(results, ["t"], [["Feature Set", "Metric", "X"]])
    assert df.shape == (16, 3)
    assert df.iloc[1, 0] == "All Features (n=169)"
    assert df.iloc[2, 0] == ""          # vertically merged label cell
    assert df.iloc[6, 0] == "Demographics Only (n=4)"
