"""SHAP analysis stratified by demographic subgroups (age, BMI).

The manuscript reports that *age dominates in younger / leaner patients
whereas BMI dominates in those with higher adiposity*.  This module
provides the helpers to reproduce that subgroup beeswarm analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..plots.style import add_panel_labels
from .compute import exclude_feature_from_shap


def _build_group_mask(
    values: np.ndarray, group_name: str, low: float, high: float,
) -> np.ndarray:
    """Match the boundary semantics used in the original notebook.

    * ``"<=X"``  group: ``values <= X``
    * ``">=X"``  group: ``values >  X``  (open lower bound)
    * intermediate: ``low < values <= high`` (right-closed)
    """
    if group_name.startswith("<="):
        return values <= high
    if group_name.startswith(">="):
        return values > low
    return (values > low) & (values <= high)


def shap_by_strata(
    shap_values: np.ndarray,
    X_display: pd.DataFrame,
    strata_col: str,
    groups: Mapping[str, Tuple[float, float]],
) -> Dict[str, Dict[str, Any]]:
    """Split SHAP values by a stratum variable (e.g. ``"Age"`` or ``"BMI"``)."""
    if strata_col not in X_display.columns:
        raise KeyError(f"Stratum column '{strata_col}' not in X_display.")

    values = X_display[strata_col].values
    out: Dict[str, Dict[str, Any]] = {}
    for group_name, (low, high) in groups.items():
        mask = _build_group_mask(values, group_name, low, high)
        out[group_name] = {
            "shap_values": shap_values[mask, :],
            "X_subset":    X_display[mask],
            "n_samples":   int(mask.sum()),
        }
    return out


def plot_shap_strata_beeswarm(
    group_shap: Dict[str, Dict[str, Any]],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
) -> None:
    """2×2 beeswarm grid — one panel per stratum group."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for idx, (group_name, data) in enumerate(group_shap.items()):
        plt.sca(axes.flatten()[idx])
        sv, xs = exclude_feature_from_shap(data["shap_values"], data["X_subset"])
        shap.summary_plot(
            sv, xs,
            max_display=max_display, show=False, plot_size=None,
            cmap="coolwarm",
            color_bar=(idx % 2 == 1),
        )
        axes.flatten()[idx].set_title(
            f"{group_name} (n = {data['n_samples']:,})", fontsize=10,
        )

    # Harmonise x-limits and tidy axis labels
    x_limits = [ax.get_xlim() for ax in axes.flatten()]
    xmin = min(lo for lo, _ in x_limits)
    xmax = max(hi for _, hi in x_limits)
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x", alpha=0.7)
        if idx < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("SHAP Value", fontsize=9)

    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    add_panel_labels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
