"""Exact (Clopper-Pearson) binomial intervals for sensitivity, specificity and precision."""

from __future__ import annotations

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact binomial interval for a proportion."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lo, hi


def threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Counts and proportions at a fixed threshold, with exact intervals."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(np.sum((p == 1) & (y == 1)))
    fn = int(np.sum((p == 0) & (y == 1)))
    tn = int(np.sum((p == 0) & (y == 0)))
    fp = int(np.sum((p == 1) & (y == 0)))
    out = {}
    for name, k, n in (("sensitivity", tp, tp + fn),
                       ("specificity", tn, tn + fp),
                       ("precision", tp, tp + fp)):
        lo, hi = clopper_pearson(k, n)
        out[name] = k / n
        out[f"{name}_lo"] = lo
        out[f"{name}_hi"] = hi
    return out


def main() -> None:
    """Write exact Clopper-Pearson intervals for every feature set and endpoint."""
    import pickle
    from pathlib import Path

    import pandas as pd

    from config import PATHS

    pkl = PATHS["xgb_results_pkl"]
    out = PATHS["tbl_clinical_utility"] / "exact_intervals" / "exact_threshold_cis.csv"
    with pkl.open("rb") as fh:
        feature_sets = pickle.load(fh)["feature_sets"]

    rows = []
    for fs, block in feature_sets.items():
        for endpoint, pred in block["all_predictions"].items():
            metrics = threshold_metrics(pred["y_true"], pred["y_pred"])
            rows.append({"feature_set": fs, "endpoint": endpoint, **metrics})
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
