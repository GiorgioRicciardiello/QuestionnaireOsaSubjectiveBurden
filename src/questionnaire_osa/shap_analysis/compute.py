"""Compute SHAP values for stored XGBoost models."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import shap


def get_model_and_test(
    feature_sets: Dict[str, Any], fs_name: str, target_name: str,
) -> Tuple[Any, pd.DataFrame, pd.Series]:
    """Pull ``(model, X_test, y_test)`` from a serialised feature-set dict."""
    block = feature_sets[fs_name]
    return (
        block["all_models"][target_name],
        block["all_test_sets"][target_name]["X_test"],
        block["all_test_sets"][target_name]["y_test"],
    )


def rename_features(
    X: pd.DataFrame, display_names: Dict[str, str],
) -> pd.DataFrame:
    """Replace raw column names with publication-ready labels."""
    return X.rename(columns=display_names)


def exclude_feature_from_shap(
    shap_values: np.ndarray,
    X_display: pd.DataFrame,
    feature: str = "Race",
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Drop a label-encoded feature from beeswarm displays.

    Race carries no ordinal interpretation, so its low/high color gradient
    in the beeswarm plot would be misleading.  The feature is *kept* in
    the underlying model and in tabular SHAP summaries — it is only
    excluded from the visual beeswarm.

    Handles both 2-D (binary) and 3-D (multiclass) SHAP arrays.
    """
    if feature not in X_display.columns:
        return shap_values, X_display

    col_idx = X_display.columns.get_loc(feature)
    keep_mask = np.ones(X_display.shape[1], dtype=bool)
    keep_mask[col_idx] = False

    if shap_values.ndim == 3:
        return shap_values[:, keep_mask, :], X_display.drop(columns=[feature])
    return shap_values[:, keep_mask], X_display.drop(columns=[feature])


def compute_shap_values(
    feature_sets: Dict[str, Any],
    fs_name: str,
    target_names: List[str],
    targets_config: Dict[str, Dict[str, Any]],
    display_names: Dict[str, str],
    sample_size: int | None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Compute :class:`shap.TreeExplainer` values for the requested targets.

    Parameters
    ----------
    feature_sets
        Sub-dict from the loaded XGBoost pickle (``artifacts["feature_sets"]``).
    fs_name
        Which feature set to operate on (e.g. ``"all_features"``).
    target_names
        Targets to compute over.
    targets_config
        Full :data:`config.TARGETS` mapping (used for task type lookup).
    display_names
        Raw → display name mapping.
    sample_size
        Number of test rows to sample.  ``None`` → use the full test set.
    random_state
        Seed for the row-sampler.

    Returns
    -------
    dict
        ``{target: {"shap_values", "X_sample", "task_type"}}``.
    """
    rng = np.random.default_rng(random_state)
    out: Dict[str, Dict[str, Any]] = {}

    for target_name in target_names:
        cfg = targets_config[target_name]
        model, X_test, _ = get_model_and_test(feature_sets, fs_name, target_name)

        if sample_size is not None and sample_size < len(X_test):
            idx = rng.choice(len(X_test), size=sample_size, replace=False)
            X_data = X_test.iloc[idx]
        else:
            X_data = X_test

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_data)
        X_display = rename_features(X_data, display_names)

        out[target_name] = {
            "shap_values": shap_vals,
            "X_sample":    X_display,
            "task_type":   cfg["type"],
        }

        if cfg["type"] == "binary":
            print(f"  [{target_name.upper()}] shape={shap_vals.shape}")
        else:
            print(
                f"  [{target_name.upper()}] shape="
                f"{shap_vals.shape[0]} samples x "
                f"{shap_vals.shape[1]} features x "
                f"{shap_vals.shape[2]} classes"
            )

    return out
