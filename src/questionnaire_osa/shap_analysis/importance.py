"""Aggregate SHAP values into ranked feature-importance tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def calculate_feature_importance_table(
    shap_values: np.ndarray,
    feature_names: List[str],
    top_n: int | None = 10,
) -> pd.DataFrame:
    """Compute mean |SHAP|, std, and signed mean per feature.

    Multiclass tensors of shape ``(n_samples, n_features, n_classes)``
    are reduced by averaging across both samples and classes.

    The output is sorted by ``Mean_|SHAP|`` (descending) and decorated
    with a ``Direction`` column for log-odds interpretation.
    """
    if shap_values.ndim == 3:                   # multiclass
        mean_abs    = np.abs(shap_values).mean(axis=(0, 2))
        std_abs     = np.abs(shap_values).std(axis=0).mean(axis=-1)
        mean_signed = shap_values.mean(axis=(0, 2))
    else:                                       # binary
        mean_abs    = np.abs(shap_values).mean(axis=0)
        std_abs     = np.abs(shap_values).std(axis=0)
        mean_signed = shap_values.mean(axis=0)

    df = pd.DataFrame({
        "Feature":          feature_names,
        "Mean_|SHAP|":      mean_abs,
        "Std":              std_abs,
        "Mean_Signed_SHAP": mean_signed,
    }).sort_values("Mean_|SHAP|", ascending=False)

    if top_n is not None:
        df = df.head(top_n)

    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Direction"] = np.where(
        df["Mean_Signed_SHAP"] > 0, "Increases odds", "Decreases odds",
    )

    return df[["Rank", "Feature", "Mean_|SHAP|", "Std", "Direction"]]


def export_importance_csv(
    shap_data_per_fs: List[Tuple[str, Dict[str, Dict[str, Any]]]],
    target_names: List[str],
    targets_config: Dict[str, Dict[str, Any]],
    output_path: Path,
    drop_columns: List[str] | None = None,
) -> pd.DataFrame:
    """Concatenate SHAP importance tables across feature sets and write CSV.

    Parameters
    ----------
    shap_data_per_fs
        List of ``(feature_set_label, shap_dict)`` pairs where
        ``shap_dict`` matches the structure returned by
        :func:`shap_analysis.compute.compute_shap_values`.
    target_names
        Targets to include (binary or multiclass — matched by caller).
    targets_config
        Full :data:`config.TARGETS` mapping for description lookup.
    output_path
        CSV destination path (parent directory must exist).
    drop_columns
        Optional columns to drop from each per-(target, fs) table — e.g.
        ``["Std", "Direction"]`` for the multiclass export.
    """
    rows: List[pd.DataFrame] = []
    drop_columns = drop_columns or []

    for fs_label, shap_dict in shap_data_per_fs:
        for target_name in target_names:
            d = shap_dict[target_name]
            tbl = calculate_feature_importance_table(
                d["shap_values"], list(d["X_sample"].columns), top_n=None,
            )
            tbl["Mean_|SHAP|"] = tbl["Mean_|SHAP|"].round(3)
            for col in drop_columns:
                if col in tbl.columns:
                    tbl = tbl.drop(columns=[col])
            tbl.insert(0, "Target",      targets_config[target_name]["desc"])
            tbl.insert(0, "Feature_Set", fs_label)
            rows.append(tbl)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(output_path, index=False)
    return out
