"""Entry point: rebuild Table 2 and its supporting statistics from a stored run.

Run with::

    python -m questionnaire_osa.manuscript.make_reference_tables [--pkl PATH]

Defaults read ``results/training/xgboost_cv_results.pkl`` (the reference run)
and write

* ``results/tables/model_performance/table2_xgboost_binary.xlsx``  — rows map 1:1 onto manuscript Table 2
* ``results/tables/model_performance/xgb_auroc_delong.csv``        — AUROC CIs + paired DeLong differences (underlies Table S9)
* ``results/tables/model_performance/xgb_wilson_ci.csv``           — Se / Sp / precision Wilson CIs (underlies Table S11)
* ``results/tables/model_performance/xgb_reference_summary.json``  — derived ranges quoted in the text
* ``results/tables/model_performance/S4_xgboost_binary.xlsx``      — Table S4
* ``results/tables/model_performance/S5_xgboost_multiclass.xlsx``  — Table S5
* ``results/tables/model_performance/S9_auroc_uncertainty.xlsx``   — Table S9
* ``results/tables/model_performance/S11_wilson_ci.xlsx``          — Table S11
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import PATHS

from ..reporting.reference_run import (
    DIFF_PAIRS, FS_DISPLAY, FS_N_FEATURES, TABLE2_FS_ORDER, build_auroc_table,
    build_table2, build_wilson_table, compute_all,
)


DEFAULT_PKL = PATHS["xgb_results_pkl"]
# TabPFN accuracy vs full-cohort majority class (5-fold CV; unaffected by the XGBoost run).
TABPFN_MAJORITY_CSV = PATHS["tbl_quantification"] / "q4_tabpfn_vs_majority.csv"
TARGET_DESC = {"ahi_5": "AHI @ 5", "ahi_15": "AHI @ 15", "ahi_30": "AHI @ 30",
               "rdi_5": "RDI @ 5", "rdi_15": "RDI @ 15", "rdi_30": "RDI @ 30"}
TABLE2_HEADER = [
    ["Feature Set", "Metric", "AHI", "AHI", "AHI", "RDI", "RDI", "RDI"],
    ["", "", "≥5", "≥15", "≥30", "≥5", "≥15", "≥30"],  # row 1 cols 0–1 are vertically merged in Word
]


def _endpoint_labels(targets_config, targets):
    return {t: targets_config[t]["desc"].replace(" @ ", " ≥") for t in targets}


def _summary(results, targets):
    """Ranges and gaps the manuscript text quotes (rounded to 3 dp)."""
    ahi = [t for t in targets if t.startswith("ahi")]
    rdi = [t for t in targets if t.startswith("rdi")]
    auc = lambda t, fs: results[t]["fs"][fs]["auroc"]

    def rng(ts, fs):
        v = [auc(t, fs) for t in ts]
        return [round(min(v), 3), round(max(v), 3)]

    out = {fs: {"ahi": rng(ahi, fs), "rdi": rng(rdi, fs), "all": rng(targets, fs)}
           for fs in TABLE2_FS_ORDER}
    out["auroc"] = {t: {fs: round(auc(t, fs), 3) for fs in TABLE2_FS_ORDER} for t in targets}
    for a, b in DIFF_PAIRS:
        d = [results[t]["diffs"][(a, b)]["delta"] for t in targets]
        lo = [results[t]["diffs"][(a, b)]["ci"][0] for t in targets]
        p = [results[t]["diffs"][(a, b)]["p"] for t in targets]
        out[f"{a}_minus_{b}"] = {
            "range": [round(min(d), 3), round(max(d), 3)],
            "ci_lower_range": [round(min(lo), 3), round(max(lo), 3)],
            "max_p": float(max(p)),
            "per_target": {t: round(x, 3) for t, x in zip(targets, d)},
        }
    # Share of the above-chance AUROC that Demographics-Only retains.
    share = [(auc(t, "dem_only") - 0.5) / (auc(t, "all_features") - 0.5) for t in targets]
    out["dem_share_above_chance_pct"] = [round(100 * min(share)), round(100 * max(share))]
    out["n_test"] = {t: results[t]["n"] for t in targets}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    args = parser.parse_args()
    if not args.pkl.exists():
        raise FileNotFoundError(f"Training artifacts not found at {args.pkl}")
    with args.pkl.open("rb") as fh:
        artifacts = pickle.load(fh)

    targets = artifacts["binary_targets"]
    labels = _endpoint_labels(artifacts["targets_config"], targets)
    results = compute_all(artifacts["feature_sets"], targets)

    # Cross-check against the metrics stored at training time.
    for t in targets:
        for fs in TABLE2_FS_ORDER:
            stored = artifacts["feature_sets"][fs]["all_results"][t]
            got = results[t]["fs"][fs]
            for key, ref in (("auroc", "auroc"), ("sensitivity", "recall"),
                             ("specificity", "specificity"), ("precision", "precision"),
                             ("f1", "f1")):
                if not np.isclose(got[key], stored[ref], atol=1e-9):
                    raise ValueError(f"{fs}/{t}/{key}: {got[key]} != stored {stored[ref]}")

    tables_main = PATHS["tbl_model_performance"]
    tables_main.mkdir(parents=True, exist_ok=True)
    build_table2(results, targets, TABLE2_HEADER).to_excel(
        tables_main / "table2_xgboost_binary.xlsx", header=False, index=False)
    with (tables_main / "xgb_reference_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(_summary(results, targets), fh, indent=2, ensure_ascii=False)

    # Supplementary tables (rows map 1:1 onto the Word tables), plus their
    # underlying raw-data CSVs kept alongside. Same task folder as Table 2
    # since all of these answer the "model performance" question.
    supp = PATHS["tbl_model_performance"]
    supp.mkdir(parents=True, exist_ok=True)
    build_auroc_table(results, targets, labels).to_csv(
        supp / "xgb_auroc_delong.csv", index=False, encoding="utf-8")
    build_wilson_table(results, targets, labels).to_csv(
        supp / "xgb_wilson_ci.csv", index=False, encoding="utf-8")
    s4_header = [["Feature Set", "Metric"] + [f"{labels[t].split(' ')[0]} {labels[t].split(' ')[1]}" for t in targets]]
    build_table2(results, targets, s4_header).to_excel(
        supp / "S4_xgboost_binary.xlsx", header=False, index=False)
    _s5(artifacts).to_excel(supp / "S5_xgboost_multiclass.xlsx", header=False, index=False)
    _s9(results, targets, labels).to_excel(supp / "S9_auroc_uncertainty.xlsx", header=False, index=False)
    s11 = build_wilson_table(results, targets, labels)
    pd.DataFrame([list(s11.columns)] + s11.values.tolist()).to_excel(
        supp / "S11_wilson_ci.xlsx", header=False, index=False)
    print(f"Tables written to {tables_main} and {supp}")


def _s5(artifacts) -> pd.DataFrame:
    rows = [["Feature Set", "AHI Macro-F1", "AHI Accuracy", "RDI Macro-F1", "RDI Accuracy"]]
    for fs in TABLE2_FS_ORDER:
        res = artifacts["feature_sets"][fs]["all_results"]
        rows.append([f"{FS_DISPLAY[fs]} (n={FS_N_FEATURES[fs]})",
                     f"{res['ahi_4class']['macro_f1']:.2f}", f"{res['ahi_4class']['accuracy']:.2f}",
                     f"{res['rdi_4class']['macro_f1']:.2f}", f"{res['rdi_4class']['accuracy']:.2f}"])
    return pd.DataFrame(rows)


def _s9(results, targets, labels) -> pd.DataFrame:
    """XGBoost DeLong columns + the unchanged TabPFN-vs-majority columns."""
    tab = pd.read_csv(TABPFN_MAJORITY_CSV).set_index("target")
    auroc = build_auroc_table(results, targets, labels)
    rows = [["Endpoint", "All-Features AUROC (95% CI)", "Demographics-Only AUROC (95% CI)",
             "Questionnaire-Only AUROC (95% CI)", "All − Demographics (paired 95% CI)",
             "All − Questionnaire (paired 95% CI)", "Majority-class accuracy",
             "TabPFN accuracy (gain over majority)"]]
    for t, (_, r) in zip(targets, auroc.iterrows()):
        b = tab.loc[TARGET_DESC[t]]
        rows.append([
            r["Endpoint"],
            r["All Features AUROC (95% CI)"], r["Demographics Only AUROC (95% CI)"],
            r["Questionnaire Only AUROC (95% CI)"],
            r["All Features − Demographics Only (paired DeLong 95% CI)"],
            r["All Features − Questionnaire Only (paired DeLong 95% CI)"],
            f"{b['majority']:.3f}", f"{b['tabpfn_accuracy']:.3f} (+{b['gain']:.3f})",
        ])
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
