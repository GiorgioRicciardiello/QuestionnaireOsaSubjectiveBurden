"""Manuscript numbers recomputed from the per-patient predictions of one run.

Everything here is derived from ``feature_sets[fs]["all_predictions"]`` of a
stored XGBoost run (held-out 10 % test set, one per endpoint):

* Table 2 — AUROC, sensitivity, specificity, precision, F1 per feature set.
* AUROC 95 % CIs and *paired* AUROC differences (DeLong, fast algorithm of
  Sun & Xu 2014), replacing the earlier independent-samples approximation.
* Wilson 95 % CIs for sensitivity, specificity and precision from the
  actual confusion-matrix counts.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score


FS_DISPLAY = {
    "all_features": "All Features",
    "dem_only":     "Demographics Only",
    "no_dem":       "Questionnaire Only",
}
FS_N_FEATURES = {"all_features": 169, "dem_only": 4, "no_dem": 161}
TABLE2_FS_ORDER = ["all_features", "dem_only", "no_dem"]
TABLE2_METRICS = ["AUROC", "Sensitivity", "Specificity", "Precision", "F1 Score"]
DIFF_PAIRS = [("all_features", "dem_only"), ("all_features", "no_dem")]
Z95 = stats.norm.ppf(0.975)


# ── DeLong ────────────────────────────────────────────────────────────────────
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong(y_true: np.ndarray, scores: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """AUROCs of several scores on the same patients and their covariance."""
    y_true = np.asarray(y_true).astype(int)
    pos, neg = y_true == 1, y_true == 0
    m, n = pos.sum(), neg.sum()
    k = len(scores)
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r, s in enumerate(scores):
        s = np.asarray(s, dtype=float)
        x, y = s[pos], s[neg]
        tx[r], ty[r] = _midrank(x), _midrank(y)
        tz[r] = _midrank(np.concatenate([x, y]))
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


# ── Wilson ────────────────────────────────────────────────────────────────────
def wilson(successes: int, total: int) -> Tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denom
    half = Z95 * np.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denom
    return centre - half, centre + half


# ── Per-endpoint computation ──────────────────────────────────────────────────
def _predictions(feature_sets: Dict[str, Any], fs: str, target: str):
    p = feature_sets[fs]["all_predictions"][target]
    return np.asarray(p["y_true"]).astype(int), np.asarray(p["y_pred"]).astype(int), np.asarray(p["y_proba"])


def endpoint_metrics(feature_sets: Dict[str, Any], target: str) -> Dict[str, Any]:
    """All Table 2 / S9 / S11 quantities for one endpoint."""
    y_ref = None
    probas, rows = [], {}
    for fs in TABLE2_FS_ORDER:
        y, yhat, proba = _predictions(feature_sets, fs, target)
        if y_ref is None:
            y_ref = y
        elif not np.array_equal(y, y_ref):
            raise ValueError(f"{target}: feature sets were not scored on the same test patients")
        tp = int(((yhat == 1) & (y == 1)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        rows[fs] = {
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": tp / (tp + fn), "sensitivity_ci": wilson(tp, tp + fn),
            "specificity": tn / (tn + fp), "specificity_ci": wilson(tn, tn + fp),
            "precision": tp / (tp + fp), "precision_ci": wilson(tp, tp + fp),
            "f1": f1_score(y, yhat),
            "auroc_sklearn": roc_auc_score(y, proba),
        }
        probas.append(proba)

    aucs, cov = delong(y_ref, probas)
    for i, fs in enumerate(TABLE2_FS_ORDER):
        se = np.sqrt(cov[i, i])
        rows[fs].update(auroc=aucs[i], auroc_ci=(aucs[i] - Z95 * se, aucs[i] + Z95 * se))

    diffs = {}
    for a, b in DIFF_PAIRS:
        i, j = TABLE2_FS_ORDER.index(a), TABLE2_FS_ORDER.index(b)
        d = aucs[i] - aucs[j]
        se = np.sqrt(cov[i, i] + cov[j, j] - 2 * cov[i, j])
        diffs[(a, b)] = {
            "delta": d, "ci": (d - Z95 * se, d + Z95 * se),
            "p": 2 * stats.norm.sf(abs(d) / se),
        }
    return {"n": len(y_ref), "n_pos": int(y_ref.sum()), "fs": rows, "diffs": diffs}


def compute_all(feature_sets: Dict[str, Any], targets: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {t: endpoint_metrics(feature_sets, t) for t in targets}


# ── Table builders (rows map 1:1 onto the Word tables) ────────────────────────
def _f3(x: float) -> str:
    return f"{x:.3f}"


def _ci(x: float, ci: Tuple[float, float]) -> str:
    return f"{x:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"


def build_table2(results: Dict[str, Dict[str, Any]], targets: Sequence[str],
                 header_rows: List[List[str]]) -> pd.DataFrame:
    keys = {"AUROC": "auroc", "Sensitivity": "sensitivity", "Specificity": "specificity",
            "Precision": "precision", "F1 Score": "f1"}
    body = []
    for fs in TABLE2_FS_ORDER:
        label = f"{FS_DISPLAY[fs]} (n={FS_N_FEATURES[fs]})"
        for i, metric in enumerate(TABLE2_METRICS):
            # Label only on the first row: the Word column is vertically merged.
            body.append([label if i == 0 else "", metric]
                        + [_f3(results[t]["fs"][fs][keys[metric]]) for t in targets])
    return pd.DataFrame(header_rows + body)


def build_auroc_table(results: Dict[str, Dict[str, Any]], targets: Sequence[str],
                      labels: Dict[str, str]) -> pd.DataFrame:
    """AUROC (DeLong CI) per feature set and paired differences."""
    rows = []
    for t in targets:
        r = results[t]
        row = {"Endpoint": labels[t], "n": r["n"], "n_positive": r["n_pos"]}
        for fs in TABLE2_FS_ORDER:
            row[f"{FS_DISPLAY[fs]} AUROC (95% CI)"] = _ci(r["fs"][fs]["auroc"], r["fs"][fs]["auroc_ci"])
        for (a, b), d in r["diffs"].items():
            name = f"{FS_DISPLAY[a]} − {FS_DISPLAY[b]}"
            row[f"{name} (paired DeLong 95% CI)"] = _ci(d["delta"], d["ci"])
            row[f"{name} p"] = f"{d['p']:.2g}" if d["p"] >= 1e-4 else "<0.0001"
        rows.append(row)
    return pd.DataFrame(rows)


def build_wilson_table(results: Dict[str, Dict[str, Any]], targets: Sequence[str],
                       labels: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for fs in TABLE2_FS_ORDER:
        for t in targets:
            m = results[t]["fs"][fs]
            rows.append({
                "Feature set": FS_DISPLAY[fs], "Endpoint": labels[t],
                "Sensitivity (95% CI)": _ci(m["sensitivity"], m["sensitivity_ci"]),
                "Specificity (95% CI)": _ci(m["specificity"], m["specificity_ci"]),
                "Precision (95% CI)": _ci(m["precision"], m["precision_ci"]),
            })
    return pd.DataFrame(rows)
