"""Classification metrics — single dispatcher + per-task helpers.

Binary metrics returned: AUROC, Accuracy, Precision, Recall, F1,
Specificity, Confusion Matrix.
Multiclass metrics: Macro-F1, Accuracy, Confusion Matrix.

A separate :func:`aggregate_metrics` collapses a list of per-fold
metric dicts into ``mean / std`` pairs for cross-validation reporting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ── Per-task helpers ──────────────────────────────────────────────────────────
def evaluate_binary(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_proba: npt.ArrayLike,
) -> Dict[str, Any]:
    """Compute the standard binary-classifier metric panel.

    Specificity is computed from the confusion matrix as
    ``TN / (TN + FP)``.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, _fn, _tp = cm.ravel()
    return {
        "auroc":       roc_auc_score(y_true, y_proba),
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "confusion_matrix": cm,
    }


def evaluate_multiclass(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
) -> Dict[str, Any]:
    """Compute Macro-F1 + Accuracy + confusion matrix for ≥3-class targets."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


# ── Dispatcher ────────────────────────────────────────────────────────────────
def evaluate(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    task_type: Literal["binary", "multiclass"] = "binary",
) -> Dict[str, Any]:
    """Convenience wrapper that routes to the binary / multiclass path."""
    if task_type == "binary":
        return evaluate_binary(y_true, y_pred, y_proba)
    return evaluate_multiclass(y_true, y_pred)


# ── CV aggregation ────────────────────────────────────────────────────────────
def aggregate_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-fold metric dicts into ``mean`` / ``std`` summaries.

    Confusion matrices are kept as a list (one per fold).  All scalar
    metrics get three derived keys: ``<m>_mean``, ``<m>_std`` and
    ``<m>_values`` (raw per-fold values).
    """
    out: Dict[str, Any] = {}
    if not fold_metrics:
        return out

    for key in fold_metrics[0]:
        if key == "confusion_matrix":
            out[key] = [m[key] for m in fold_metrics]
            continue
        values = [m[key] for m in fold_metrics]
        out[f"{key}_mean"]   = float(np.mean(values))
        out[f"{key}_std"]    = float(np.std(values))
        out[f"{key}_values"] = values
    return out


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    """Render ``mean ± std`` with consistent precision for tables."""
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
