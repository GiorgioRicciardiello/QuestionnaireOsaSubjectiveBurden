"""TabPFN v2.5 cross-validation runner.

Unlike the XGBoost pipeline, TabPFN is hyper-parameter-free and only
requires a single forward pass per fold.  We therefore evaluate it with
a 5-fold multilabel-stratified CV (no held-out test set) and report
``mean ± std`` of every metric across folds.

Hardware note
-------------
The model is a transformer; CPU inference on 60k samples is intractable.
The default device is ``"cuda"`` (see :data:`config.SETTINGS`).
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..evaluation.metrics import (
    aggregate_metrics,
    evaluate_binary,
    evaluate_multiclass,
)
from ..stratification.splitter import get_stratified_kfold


def run_cv_experiment_tabpfn(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    target_type: str,
    stratification_config: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    classification_threshold: float = 0.5,
    device: str = "cuda",
    random_state: int = 42,
) -> Tuple[
    List[Dict[str, Any]],
    List[Tuple[Any, ...]],
    List[Tuple[np.ndarray, np.ndarray]],
]:
    """Run *n_splits*-fold CV for one target with TabPFN v2.5.

    Returns
    -------
    fold_metrics
        Per-fold metric dictionaries.
    fold_predictions
        Per-fold ``(y_true, y_pred, y_proba)`` tuples.
    fold_indices
        Per-fold ``(train_idx, test_idx)`` pandas-index arrays — useful
        for downstream stratification verification.
    """
    # Imports kept lazy: TabPFN brings PyTorch + GPU drivers into scope.
    import torch
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion

    kfold, strat_matrix = get_stratified_kfold(
        df, y, stratification_config, n_splits=n_splits, random_state=random_state,
    )

    fold_metrics:     List[Dict[str, Any]]              = []
    fold_predictions: List[Tuple[Any, ...]]              = []
    fold_indices:     List[Tuple[np.ndarray, np.ndarray]] = []

    # v2.5 explicitly: v2.6 hangs on preprocessing for our N
    model = TabPFNClassifier.create_default_for_version(
        ModelVersion.V2_5, device=device, ignore_pretraining_limits=True,
    )

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X.values, strat_matrix)):
        t0 = time.time()
        fold_indices.append(
            (X.index[train_idx].values, X.index[test_idx].values)
        )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)

        if target_type == "binary":
            y_pred = (y_proba[:, 1] >= classification_threshold).astype(int)
            metrics = evaluate_binary(y_test, y_pred, y_proba[:, 1])
            fold_predictions.append((y_test.values, y_pred, y_proba[:, 1]))
            primary = f"AUROC: {metrics['auroc']:.4f}"
        else:
            y_pred = y_proba.argmax(axis=1)
            metrics = evaluate_multiclass(y_test, y_pred)
            fold_predictions.append((y_test.values, y_pred, y_proba))
            primary = f"Macro F1: {metrics['macro_f1']:.4f}"

        fold_metrics.append(metrics)
        print(f"    Fold {fold + 1}/{n_splits}: {primary}  ({time.time() - t0:.0f}s)")

    # Free GPU memory between targets
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return fold_metrics, fold_predictions, fold_indices


def run_all_targets_tabpfn(
    X: pd.DataFrame,
    df: pd.DataFrame,
    targets: Dict[str, Dict[str, Any]],
    stratification_config: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    classification_threshold: float = 0.5,
    device: str = "cuda",
    random_state: int = 42,
) -> Dict[str, Any]:
    """Run TabPFN CV for every target.  Returns aggregated artefacts."""
    all_results:     Dict[str, Dict[str, Any]] = {}
    all_predictions: Dict[str, List[Tuple[Any, ...]]] = {}
    all_indices:     Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    for idx, (target_name, cfg) in enumerate(targets.items(), 1):
        print(f"\n[{idx}/{len(targets)}] {target_name}: {cfg['desc']}")
        print("-" * 60)
        y = df[cfg["col"]]

        fold_metrics, fold_preds, fold_idx = run_cv_experiment_tabpfn(
            X, y, df, cfg["type"], stratification_config,
            n_splits=n_splits,
            classification_threshold=classification_threshold,
            device=device, random_state=random_state,
        )

        all_results[target_name]     = aggregate_metrics(fold_metrics)
        all_predictions[target_name] = fold_preds
        all_indices[target_name]     = fold_idx

    return {
        "all_results":     all_results,
        "all_predictions": all_predictions,
        "all_fold_indices": all_indices,
    }
