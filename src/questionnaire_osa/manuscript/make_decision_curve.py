"""Entry point: decision curve net benefit for the stored XGBoost run.

Run with::

    python -m questionnaire_osa.manuscript.make_decision_curve [--pkl PATH] [--out-dir DIR]
                                                    [--resamples N] [--seed S]

Reads the held-out test-set predictions in
``results/training/xgboost_cv_results.pkl``
(``feature_sets[fs]["all_predictions"][endpoint]``), so no model is retrained.
Writes to ``results/tables/clinical_utility/decision_curve/``:

* ``calibration_metrics.csv``: calibration slope and intercept, mean predicted
  probability, prevalence and Brier score per endpoint and feature set.
  (Supplementary Table S13.)
* ``dca_netbenefit_curves.csv``: net benefit for each feature set and endpoint
  on the 0.01 to 0.80 grid, with treat-all and treat-none references.
  (Underlying data for Supplementary Figure S15.)
* ``dca_delta_nb_bootstrap.csv``: All-Features minus Demographics-Only and
  All-Features minus Questionnaire-Only, paired stratified bootstrap at the
  anchor thresholds. (Supplementary Table S12.)

and to ``results/figures/clinical_utility/``:

* ``figureS15_decision_curves.{png,pdf}``

Probabilities are the uncalibrated XGBoost outputs. Class weighting shifts
them (see docs/net_benefit_plan.md), so results are provisional until
recalibration is done.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS

from ..plots.decision_curve_figure import plot_decision_curves
from ..evaluation.decision_curve import (
    DEFAULT_ANCHORS,
    DEFAULT_THRESHOLDS,
    calibration_summary,
    decision_curve_frame,
    delta_net_benefit_bootstrap,
)

DEFAULT_PKL = PATHS["xgb_results_pkl"]
DEFAULT_OUT_DIR = PATHS["tbl_clinical_utility"] / "decision_curve"
DEFAULT_FIG_DIR = PATHS["fig_clinical_utility"]
ENDPOINTS = ("ahi_5", "ahi_15", "ahi_30", "rdi_5", "rdi_15", "rdi_30")
FEATURE_SETS = {
    "all_features": "All-Features",
    "dem_only": "Demographics-Only",
    "no_dem": "Questionnaire-Only",
}
CONTRASTS = (
    ("all_features", "dem_only"),
    ("all_features", "no_dem"),
)


def _load_predictions(pkl: Path) -> dict:
    if not pkl.exists():
        raise FileNotFoundError(f"Training artifacts not found at {pkl}")
    with pkl.open("rb") as fh:
        return pickle.load(fh)["feature_sets"]


def _endpoint_predictions(feature_sets: dict, endpoint: str) -> tuple[np.ndarray, dict]:
    """Return the shared outcome vector and each feature set's probabilities.

    Raises if the feature sets do not share the same test patients, because a
    paired comparison is only valid on identical patients.
    """
    truth = None
    probs: dict[str, np.ndarray] = {}
    for fs in FEATURE_SETS:
        block = feature_sets[fs]["all_predictions"][endpoint]
        y = np.asarray(block["y_true"]).astype(int)
        if truth is None:
            truth = y
        elif not np.array_equal(truth, y):
            raise ValueError(f"{endpoint}: {fs} test patients differ from all_features")
        probs[fs] = np.asarray(block["y_proba"], dtype=float)
    return truth, probs


def build_calibration(feature_sets: dict) -> pd.DataFrame:
    rows = []
    for endpoint in ENDPOINTS:
        _, probs = _endpoint_predictions(feature_sets, endpoint)
        y, _ = _endpoint_predictions(feature_sets, endpoint)
        for fs, label in FEATURE_SETS.items():
            rows.append({"endpoint": endpoint, "strategy": label,
                         **calibration_summary(y, probs[fs])})
    return pd.DataFrame(rows)


def build_tables(
    feature_sets: dict, *, n_resamples: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_parts, boot_parts = [], []
    for endpoint in ENDPOINTS:
        y, probs = _endpoint_predictions(feature_sets, endpoint)
        labelled = {FEATURE_SETS[fs]: probs[fs] for fs in FEATURE_SETS}
        curves = decision_curve_frame(y, labelled, thresholds=DEFAULT_THRESHOLDS)
        curve_parts.append(curves.assign(endpoint=endpoint))

        for a, b in CONTRASTS:
            boot = delta_net_benefit_bootstrap(
                y, probs[a], probs[b], thresholds=DEFAULT_ANCHORS,
                n_resamples=n_resamples, random_state=seed,
            )
            boot_parts.append(boot.assign(
                endpoint=endpoint,
                contrast=f"{FEATURE_SETS[a]} minus {FEATURE_SETS[b]}",
            ))

    curves_all = pd.concat(curve_parts, ignore_index=True)[
        ["endpoint", "strategy", "threshold", "net_benefit"]
    ]
    boot_all = pd.concat(boot_parts, ignore_index=True)[
        ["endpoint", "contrast", "threshold", "delta_nb", "ci_lo", "ci_hi"]
    ]
    return curves_all, boot_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    feature_sets = _load_predictions(args.pkl)
    curves, boot = build_tables(feature_sets, n_resamples=args.resamples, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    curves_path = args.out_dir / "dca_netbenefit_curves.csv"
    boot_path = args.out_dir / "dca_delta_nb_bootstrap.csv"
    curves.to_csv(curves_path, index=False)
    boot.to_csv(boot_path, index=False)
    calib_path = args.out_dir / "calibration_metrics.csv"
    build_calibration(feature_sets).to_csv(calib_path, index=False)
    print(f"Saved: {calib_path}")
    print(f"Saved: {curves_path}")
    print(f"Saved: {boot_path}")

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig_stem = args.fig_dir / "figureS15_decision_curves"
    for path in plot_decision_curves(curves, boot, fig_stem):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
