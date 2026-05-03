"""XGBoost training pipeline with Optuna hyper-parameter search.

Each ``(target, feature-set)`` cell is fit with a 90/10 multilabel-stratified
train/test split, followed by *n_trials* of Optuna search over a 5-fold
multilabel-stratified CV inside the train portion.  The final model is
then retrained on the full train portion using the best hyper-parameters
and evaluated once on the held-out test set.

Search direction is ``maximize`` (AUROC for binary, Macro-F1 for multiclass).
Optuna's TPE sampler is seeded for full reproducibility.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import numpy as np
import numpy.typing as npt
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, roc_auc_score

from ..evaluation.metrics import evaluate
from ..stratification.splitter import (
    get_stratified_kfold,
    stratify_train_test,
)


# ── Optuna objective ──────────────────────────────────────────────────────────
def _suggest_xgb_params(
    trial: optuna.Trial, random_state: int,
) -> Dict[str, Any]:
    """Return the XGBoost hyper-parameter dict for a single Optuna trial."""
    return {
        "max_depth":         trial.suggest_int  ("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators":      trial.suggest_int  ("n_estimators", 100, 1000),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight":  trial.suggest_int  ("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0, 5),
        "reg_alpha":         trial.suggest_float("reg_alpha",  1e-8, 10, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "tree_method":       "hist",
        "device":            "cpu",
        "random_state":      random_state,
    }


def _attach_task_params(
    params: Dict[str, Any],
    task_type: Literal["binary", "multiclass"],
    scale_pos_weight: float | None,
    num_classes: int,
) -> Dict[str, Any]:
    """Add task-specific keys (objective, num_class, eval_metric, ...)."""
    if task_type == "binary":
        params["scale_pos_weight"] = scale_pos_weight
        params["eval_metric"] = "auc"
    else:
        params["objective"]   = "multi:softprob"
        params["num_class"]   = num_classes
        params["eval_metric"] = "mlogloss"
    return params


def _objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    stratification_config: Dict[str, Dict[str, Any]],
    task_type: Literal["binary", "multiclass"],
    scale_pos_weight: float | None,
    num_classes: int,
    n_folds: int,
    random_state: int,
) -> float:
    """Mean CV score across *n_folds* for a single Optuna trial."""
    params = _attach_task_params(
        _suggest_xgb_params(trial, random_state),
        task_type, scale_pos_weight, num_classes,
    )

    kfold, strat_matrix = get_stratified_kfold(
        df, y, stratification_config, n_folds, random_state,
    )
    scores: list[float] = []

    for train_idx, val_idx in kfold.split(X.values, strat_matrix):
        model = xgb.XGBClassifier(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)

        if task_type == "binary":
            y_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(float(roc_auc_score(y.iloc[val_idx], y_proba)))
        else:
            y_pred = model.predict(X.iloc[val_idx])
            scores.append(float(f1_score(y.iloc[val_idx], y_pred, average="macro")))

    return float(np.mean(scores))


# ── Final-model fit ───────────────────────────────────────────────────────────
def _fit_final(
    X: pd.DataFrame,
    y: pd.Series,
    best_params: Dict[str, Any],
    task_type: Literal["binary", "multiclass"],
    scale_pos_weight: float | None,
    num_classes: int,
    random_state: int,
) -> xgb.XGBClassifier:
    """Refit the best hyper-parameter set on the full train portion."""
    params: Dict[str, Any] = {
        **best_params,
        "tree_method":  "hist",
        "device":       "cpu",
        "random_state": random_state,
    }
    params = _attach_task_params(params, task_type, scale_pos_weight, num_classes)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, verbose=False)
    return model


# ── Public API ────────────────────────────────────────────────────────────────
def train_with_optuna_cv(
    X: pd.DataFrame,
    y: pd.Series,
    df: pd.DataFrame,
    stratification_config: Dict[str, Dict[str, Any]],
    task_type: Literal["binary", "multiclass"] = "binary",
    num_classes: int = 4,
    n_trials: int = 100,
    n_folds: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[xgb.XGBClassifier, Dict[str, Any], optuna.Study]:
    """Run Optuna search → refit best → return ``(model, best_params, study)``.

    Parameters
    ----------
    X, y, df
        Train portion (model input, target, full row context for stratification).
    stratification_config
        Same shape as :data:`config.STRATIFICATION_CONFIG`.
    task_type
        ``"binary"`` (AUROC) or ``"multiclass"`` (Macro-F1).
    """
    scale_pos_weight = (
        float((y == 0).sum()) / float((y == 1).sum())
        if task_type == "binary" else None
    )
    metric_name = "AUROC" if task_type == "binary" else "Macro F1"

    if verbose:
        print(f"  Optuna search [{task_type}] — N={len(X):,}, optimizing {metric_name}")
    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: _objective(
            trial, X, y, df, stratification_config,
            task_type, scale_pos_weight, num_classes, n_folds, random_state,
        ),
        n_trials=n_trials,
        show_progress_bar=verbose,
    )

    if verbose:
        print(f"  Best CV {metric_name}: {study.best_value:.4f}")

    model = _fit_final(
        X, y, study.best_params, task_type,
        scale_pos_weight, num_classes, random_state,
    )
    return model, study.best_params, study


def train_all_targets_xgb(
    X: pd.DataFrame,
    df: pd.DataFrame,
    targets: Dict[str, Dict[str, Any]],
    stratification_config: Dict[str, Dict[str, Any]],
    train_pct: int = 90,
    val_pct: int = 10,
    classification_threshold: float = 0.5,
    n_trials: int = 100,
    n_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train one XGBoost model per target for a fixed feature set.

    Returns a dict with five aligned sub-dicts (keyed by target name):
    ``all_models``, ``all_results``, ``all_best_params``,
    ``all_predictions``, ``all_test_sets``.
    """
    out_models: Dict[str, Any] = {}
    out_results: Dict[str, Any] = {}
    out_params:  Dict[str, Any] = {}
    out_preds:   Dict[str, Any] = {}
    out_tests:   Dict[str, Any] = {}

    for target_name, cfg in targets.items():
        task_type = cfg["type"]
        print(f"\n[{target_name.upper()}] {cfg['desc']}")
        print("-" * 60)

        y = df[cfg["col"]]

        # Stratified outer split (train/val vs held-out test)
        (X_tv, y_tv), (X_te, y_te) = stratify_train_test(
            df, X, y, train_pct, val_pct, stratification_config, random_state,
        )
        df_tv = df.loc[X_tv.index]
        out_tests[target_name] = {"X_test": X_te, "y_test": y_te}
        print(f"  Split: train_val={len(X_tv):,}  test={len(X_te):,}")

        model, best_params, _study = train_with_optuna_cv(
            X_tv, y_tv, df_tv, stratification_config,
            task_type=task_type,
            n_trials=n_trials, n_folds=n_folds, random_state=random_state,
        )

        # Evaluate on the held-out test set
        y_proba = model.predict_proba(X_te)
        if task_type == "binary":
            y_pred = (y_proba[:, 1] >= classification_threshold).astype(int)
            metrics = evaluate(y_te, y_pred, y_proba[:, 1], "binary")
            out_preds[target_name] = {
                "y_true":  y_te.values,
                "y_pred":  y_pred,
                "y_proba": y_proba[:, 1],
            }
            print(f"  Test AUROC: {metrics['auroc']:.4f}  F1: {metrics['f1']:.4f}")
        else:
            y_pred = y_proba.argmax(axis=1)
            metrics = evaluate(y_te, y_pred, y_proba, "multiclass")
            out_preds[target_name] = {
                "y_true":  y_te.values,
                "y_pred":  y_pred,
                "y_proba": y_proba,
            }
            print(f"  Test Macro F1: {metrics['macro_f1']:.4f}  Acc: {metrics['accuracy']:.4f}")

        out_models[target_name]  = model
        out_results[target_name] = metrics
        out_params[target_name]  = best_params

    return {
        "all_models":      out_models,
        "all_results":     out_results,
        "all_best_params": out_params,
        "all_predictions": out_preds,
        "all_test_sets":   out_tests,
    }
