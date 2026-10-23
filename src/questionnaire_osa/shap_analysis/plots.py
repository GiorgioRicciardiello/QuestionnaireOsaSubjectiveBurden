"""SHAP beeswarm visualisations (binary + multiclass)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.ticker import MultipleLocator

from ..plots.style import SHAP_CMAP, add_panel_labels
from .annotate import (
    FIG_W, PT_PANEL, PT_SUPTITLE, add_note, class_title, finish_beeswarm,
    label_mean_abs,
)
from .compute import exclude_feature_from_shap


# ── Beeswarm: binary targets ──────────────────────────────────────────────────
def _harmonize_xlims_and_grid(
    axes: np.ndarray, n_top_row: int, share_x: bool = True,
) -> None:
    """Share x-limits across panels (optional) and clean up labels."""
    x_limits = [ax.get_xlim() for ax in axes.flatten()]
    xmin = min(lo for lo, _ in x_limits)
    xmax = max(hi for _, hi in x_limits)
    for idx, ax in enumerate(axes.flatten()):
        if share_x:
            ax.set_xlim(xmin, xmax)
            ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.grid(axis="x", alpha=0.7)
        if idx < n_top_row:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("SHAP Value")


def plot_shap_beeswarm_binary(
    shap_data: Dict[str, Dict[str, Any]],
    binary_targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
) -> None:
    """2×3 grid of SHAP beeswarms — one per binary target."""
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, 6.0))

    for idx, target_name in enumerate(binary_targets):
        plt.sca(axes.flatten()[idx])
        d = shap_data[target_name]
        sv, xs = exclude_feature_from_shap(d["shap_values"], d["X_sample"])
        shap.summary_plot(
            sv, xs,
            max_display=max_display, show=False, plot_size=None,
            cmap=SHAP_CMAP,
            color_bar=(idx % 3 == 2),
        )
        desc = targets_config[target_name]["desc"].replace(" @ ", " ≥ ")
        axes.flatten()[idx].set_title(desc)
        label_mean_abs(axes.flatten()[idx], sv, xs)

    _harmonize_xlims_and_grid(axes, n_top_row=3)
    plt.suptitle(suptitle, fontsize=PT_SUPTITLE, fontweight="bold")
    finish_beeswarm(fig, axes)
    add_panel_labels(axes, x=-0.02, y=1.0, fontsize=PT_PANEL, va="bottom")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=1.2, w_pad=0.6)
    add_note(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Beeswarm: multiclass targets ──────────────────────────────────────────────
def plot_shap_beeswarm_multiclass(
    shap_data: Dict[str, Dict[str, Any]],
    target_name: str,
    class_names: Sequence[str],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
) -> None:
    """2×2 grid: one beeswarm per class for a multiclass target."""
    d = shap_data[target_name]
    shap_values, X_sample = exclude_feature_from_shap(
        d["shap_values"], d["X_sample"],
    )

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.8))

    for class_idx in range(len(class_names)):
        plt.sca(axes.flatten()[class_idx])
        class_shap = shap_values[:, :, class_idx]
        shap.summary_plot(
            class_shap, X_sample,
            max_display=max_display, show=False, plot_size=None,
            cmap=SHAP_CMAP,
            color_bar=(class_idx % 2 == 1),
        )
        axes.flatten()[class_idx].set_title(
            class_title(target_name.split("_")[0].upper(), class_idx)
        )
        label_mean_abs(axes.flatten()[class_idx], class_shap, X_sample)

    # Per-class scales: class contributions differ ~10x (e.g. Mild vs No OSA),
    # a shared axis would flatten the weaker classes.
    _harmonize_xlims_and_grid(axes, n_top_row=2, share_x=False)
    plt.suptitle(suptitle, fontsize=PT_SUPTITLE, fontweight="bold")
    finish_beeswarm(fig, axes)
    add_panel_labels(axes, x=-0.02, y=1.0, fontsize=PT_PANEL, va="bottom")
    plt.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=1.2, w_pad=0.6)
    add_note(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
