"""Decision curve analysis: net benefit against threshold probability.

Net benefit (Vickers & Elkin, 2006) for a strategy that refers every patient
with predicted probability >= ``pt`` for testing:

    NB(pt) = TP/n - FP/n * pt / (1 - pt)

The odds term converts false positives into true-positive equivalents at the
stated threshold. Treat-all and treat-none are the reference strategies.
Pure functions only: no input array is modified.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: np.ndarray = np.round(np.arange(1, 81) / 100, 2)  # 0.01 .. 0.80
DEFAULT_ANCHORS: tuple[float, ...] = (0.10, 0.20, 0.30)


def _check_threshold(pt: float) -> None:
    if not 0.0 < pt < 1.0:
        raise ValueError(f"threshold_prob must be in (0, 1), got {pt}")


def net_benefit(y_true: np.ndarray, prob: np.ndarray, pt: float) -> float:
    """Net benefit of referring patients with ``prob >= pt``."""
    _check_threshold(pt)
    y = np.asarray(y_true)
    p = np.asarray(prob, dtype=float)
    n = y.size
    flagged = p >= pt
    tp = np.count_nonzero(flagged & (y == 1))
    fp = np.count_nonzero(flagged & (y == 0))
    return float(tp / n - fp / n * pt / (1.0 - pt))


def treat_all_net_benefit(y_true: np.ndarray, pt: float) -> float:
    """Net benefit of referring every patient (reference strategy)."""
    _check_threshold(pt)
    y = np.asarray(y_true)
    n = y.size
    tp = np.count_nonzero(y == 1)
    fp = np.count_nonzero(y == 0)
    return float(tp / n - fp / n * pt / (1.0 - pt))


def net_benefit_curve(
    y_true: np.ndarray,
    prob: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Net benefit of one model across a grid of thresholds."""
    rows = [
        {"threshold": float(pt), "net_benefit": net_benefit(y_true, prob, pt)}
        for pt in thresholds
    ]
    return pd.DataFrame(rows, columns=["threshold", "net_benefit"])


def decision_curve_frame(
    y_true: np.ndarray,
    probs: Mapping[str, np.ndarray],
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> pd.DataFrame:
    """Long-format decision curve: every model plus treat-all and treat-none.

    Returns columns ``threshold``, ``strategy``, ``net_benefit``.
    """
    frames = []
    for name, prob in probs.items():
        curve = net_benefit_curve(y_true, prob, thresholds)
        frames.append(curve.assign(strategy=name))
    grid = list(thresholds)
    frames.append(pd.DataFrame({
        "threshold": grid,
        "net_benefit": [treat_all_net_benefit(y_true, pt) for pt in grid],
        "strategy": "treat_all",
    }))
    frames.append(pd.DataFrame({
        "threshold": grid, "net_benefit": 0.0, "strategy": "treat_none",
    }))
    return pd.concat(frames, ignore_index=True)[["threshold", "strategy", "net_benefit"]]


def delta_net_benefit_bootstrap(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_ANCHORS,
    *,
    n_resamples: int = 1000,
    random_state: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Paired difference NB(a) - NB(b) at each threshold, with a percentile CI.

    Resampling is stratified by outcome so every resample keeps the case count
    fixed, and the same patient indices are used for both models (paired).
    Only the pre-specified ``thresholds`` are tested, never the dense grid.
    """
    y = np.asarray(y_true)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(random_state)

    rows = []
    for pt in thresholds:
        estimate = net_benefit(y, a, pt) - net_benefit(y, b, pt)
        deltas = np.empty(n_resamples)
        for i in range(n_resamples):
            idx = np.concatenate([
                rng.choice(pos, pos.size, replace=True),
                rng.choice(neg, neg.size, replace=True),
            ])
            deltas[i] = net_benefit(y[idx], a[idx], pt) - net_benefit(y[idx], b[idx], pt)
        rows.append({
            "threshold": float(pt),
            "delta_nb": estimate,
            "ci_lo": float(np.percentile(deltas, 100 * alpha / 2)),
            "ci_hi": float(np.percentile(deltas, 100 * (1 - alpha / 2))),
        })
    return pd.DataFrame(rows, columns=["threshold", "delta_nb", "ci_lo", "ci_hi"])


def calibration_summary(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Calibration-in-the-large and slope from a logistic recalibration model.

    Fits logit(P(y=1)) = intercept + slope * logit(prob). Slope near 1 and
    intercept near 0 indicate calibrated probabilities. Returns plain values.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e9, max_iter=1000).fit(logit, y)
    return {
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "calibration_slope": float(model.coef_[0][0]),
        "calibration_intercept": float(model.intercept_[0]),
        "brier": float(np.mean((p - y) ** 2)),
    }
