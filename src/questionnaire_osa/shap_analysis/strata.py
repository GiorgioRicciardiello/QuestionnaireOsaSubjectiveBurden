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
from matplotlib.ticker import MultipleLocator

from .annotate import (
    FIG_W, PT_PANEL, PT_SUPTITLE, add_note, finish_beeswarm, label_mean_abs,
)
from ..plots.style import SHAP_CMAP, add_panel_labels
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


def display_group_label(group_name: str) -> str:
    """Typeset a config group key exactly as ``_build_group_mask`` selects it.

    ``"<=36"`` -> ``≤ 36``; ``">=58"`` -> ``> 58`` (open lower bound);
    ``"36-46"`` -> ``> 36 to ≤ 46`` (right-closed).
    """
    if group_name.startswith("<="):
        return f"≤ {group_name[2:]}"
    if group_name.startswith(">="):
        return f"> {group_name[2:]}"
    low, high = group_name.split("-")
    return f"> {low} to ≤ {high}"


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
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.0))

    for idx, (group_name, data) in enumerate(group_shap.items()):
        plt.sca(axes.flatten()[idx])
        sv, xs = exclude_feature_from_shap(data["shap_values"], data["X_subset"])
        shap.summary_plot(
            sv, xs,
            max_display=max_display, show=False, plot_size=None,
            cmap=SHAP_CMAP,
            color_bar=(idx % 2 == 1),
        )
        axes.flatten()[idx].set_title(
            f"{display_group_label(group_name)} (n = {data['n_samples']:,})",
        )
        label_mean_abs(axes.flatten()[idx], sv, xs)

    # Harmonise x-limits and tidy axis labels
    x_limits = [ax.get_xlim() for ax in axes.flatten()]
    xmin = min(lo for lo, _ in x_limits)
    xmax = max(hi for _, hi in x_limits)
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlim(xmin, xmax)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(axis="x", alpha=0.7)
        if idx < 2:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("SHAP Value")

    plt.suptitle(suptitle, fontsize=PT_SUPTITLE, fontweight="bold")
    finish_beeswarm(fig, axes)
    add_panel_labels(axes, x=-0.02, y=1.0, fontsize=PT_PANEL, va="bottom")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=1.2, w_pad=0.6)
    add_note(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
