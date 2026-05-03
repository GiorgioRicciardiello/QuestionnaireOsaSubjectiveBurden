"""Multilabel stratified splitting utilities.

The cohort is jointly stratified on demographic (age, gender, BMI, race),
sleep-symptom (snore, witnessed apnea), and recording-site (lab,
physician) variables in addition to the target itself.  This guarantees
that every fold and the held-out test set carry the same case-mix as
the full cohort.

The two public entry points are:

* :func:`stratify_train_test` — single 90/10 (or arbitrary) split
* :func:`get_stratified_kfold` — k-fold CV iterator
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit,
)


# ── Internal helpers ──────────────────────────────────────────────────────────
def _bin_continuous_labels(bins: List[float]) -> List[str]:
    """Construct human-readable labels for ``pd.cut`` bin edges."""
    return (
        [f"<{bins[1]}"]
        + [f"{bins[i]}-{bins[i + 1]}" for i in range(1, len(bins) - 2)]
        + [f">={bins[-2]}"]
    )


# ── Public API ────────────────────────────────────────────────────────────────
def create_stratification_matrix(
    df: pd.DataFrame,
    stratification_features: Dict[str, Dict[str, Any]],
) -> npt.NDArray[np.int_]:
    """Build a binary one-hot matrix combining all stratification variables.

    Continuous variables are first discretised with ``pd.cut``; categorical
    variables have ``NaN`` filled with ``-1`` so that missingness itself
    is balanced across folds.

    Parameters
    ----------
    df
        Source dataframe.
    stratification_features
        Mapping ``{column: {"type": "continuous"|"categorical",
        "bins": list[float] (continuous only)}}``.

    Returns
    -------
    numpy.ndarray
        Concatenated one-hot matrix of shape ``(n_samples, n_indicators)``.
    """
    blocks: List[npt.NDArray[np.int_]] = []

    for feature_name, cfg in stratification_features.items():
        if cfg["type"] == "continuous":
            labels = _bin_continuous_labels(cfg["bins"])
            binned = pd.cut(df[feature_name], bins=cfg["bins"], labels=labels)
            blocks.append(pd.get_dummies(binned, prefix=feature_name).values)
        else:
            cat_data = df[feature_name].fillna(-1).astype(int).astype(str)
            blocks.append(pd.get_dummies(cat_data, prefix=feature_name).values)

    return np.hstack(blocks)


def _augment_with_target(
    strat_matrix: npt.NDArray[np.int_], y: pd.Series,
) -> npt.NDArray[np.int_]:
    """Append a one-hot encoding of *y* to *strat_matrix*."""
    y_encoded = pd.get_dummies(y, prefix="target").values
    return np.hstack([strat_matrix, y_encoded])


def stratify_train_test(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    train_pct: int,
    val_pct: int,
    stratification_features: Dict[str, Dict[str, Any]],
    random_state: int = 42,
) -> Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]:
    """Single multilabel-stratified train/test split.

    Parameters
    ----------
    df, X, y
        Aligned dataframes / series — ``X`` and ``y`` must share *df*'s index.
    train_pct, val_pct
        Integer percentages (e.g. ``90, 10``).  Used only as a ratio.
    stratification_features
        Same shape as in :func:`create_stratification_matrix`.
    random_state
        Seed forwarded to :class:`MultilabelStratifiedShuffleSplit`.

    Returns
    -------
    ``((X_train, y_train), (X_test, y_test))``
    """
    strat_matrix = create_stratification_matrix(df, stratification_features)
    strat_matrix = _augment_with_target(strat_matrix, y)

    test_size = val_pct / (train_pct + val_pct)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state,
    )
    train_idx, test_idx = next(msss.split(X.values, strat_matrix))

    return (
        (X.iloc[train_idx], y.iloc[train_idx]),
        (X.iloc[test_idx],  y.iloc[test_idx]),
    )


def get_stratified_kfold(
    df: pd.DataFrame,
    y: pd.Series,
    stratification_features: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[MultilabelStratifiedKFold, npt.NDArray[np.int_]]:
    """Return ``(kfold_iterator, stratification_matrix)`` ready for ``.split``.

    The caller is expected to do::

        kfold, strat = get_stratified_kfold(...)
        for train_idx, val_idx in kfold.split(X.values, strat):
            ...
    """
    strat_matrix = create_stratification_matrix(df, stratification_features)
    strat_matrix = _augment_with_target(strat_matrix, y)

    kfold = MultilabelStratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state,
    )
    return kfold, strat_matrix
