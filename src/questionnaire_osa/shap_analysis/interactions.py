"""Rank feature pairs by mean absolute SHAP interaction strength."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import shap

N_INTERACTION_SAMPLES = 1000


def rank_interactions(
    model: Any,
    X_test: pd.DataFrame,
    display_names: Dict[str, str],
    random_state: int = 42,
    top_k: int = 30,
) -> pd.DataFrame:
    """Top ``top_k`` feature pairs by mean |SHAP interaction| (log-odds).

    Computed on a random subsample, since interaction values scale with
    n_features² per row.  Strength is the off-diagonal entry summed over
    both triangles (i.e. the full pairwise effect).
    """
    X = X_test.sample(min(N_INTERACTION_SAMPLES, len(X_test)), random_state=random_state)
    inter = shap.TreeExplainer(model).shap_interaction_values(X)
    strength = 2 * np.abs(inter).mean(axis=0)
    names = [display_names.get(c, c) for c in X.columns]
    rows = [
        (names[i], names[j], strength[i, j])
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    out = pd.DataFrame(rows, columns=["Feature A", "Feature B", "Mean |SHAP interaction|"])
    out = out.sort_values("Mean |SHAP interaction|", ascending=False).head(top_k)
    out.insert(0, "Rank", range(1, len(out) + 1))
    out["n_samples"] = len(X)
    return out.reset_index(drop=True)
