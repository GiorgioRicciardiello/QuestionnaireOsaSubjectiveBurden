"""SHAP visualisations: beeswarm (binary + multiclass) and 3×3 dependence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..plots.style import add_panel_labels
from .compute import exclude_feature_from_shap


# ── Beeswarm: binary targets ──────────────────────────────────────────────────
def _harmonize_xlims_and_grid(axes: np.ndarray, n_top_row: int) -> None:
    """Share x-limits across panels and clean up labels."""
    x_limits = [ax.get_xlim() for ax in axes.flatten()]
    xmin = min(lo for lo, _ in x_limits)
    xmax = max(hi for _, hi in x_limits)
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlim(xmin, xmax)
        ax.grid(axis="x", alpha=0.7)
        if idx < n_top_row:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("SHAP Value", fontsize=9)


def plot_shap_beeswarm_binary(
    shap_data: Dict[str, Dict[str, Any]],
    binary_targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
) -> None:
    """2×3 grid of SHAP beeswarms — one per binary target."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))

    for idx, target_name in enumerate(binary_targets):
        plt.sca(axes.flatten()[idx])
        d = shap_data[target_name]
        sv, xs = exclude_feature_from_shap(d["shap_values"], d["X_sample"])
        shap.summary_plot(
            sv, xs,
            max_display=max_display, show=False, plot_size=None,
            cmap="coolwarm",
            color_bar=(idx % 3 == 2),
        )
        desc = targets_config[target_name]["desc"].replace(" @ ", " ≥ ")
        axes.flatten()[idx].set_title(desc, fontsize=10)

    _harmonize_xlims_and_grid(axes, n_top_row=3)
    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    add_panel_labels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
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

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for class_idx in range(len(class_names)):
        plt.sca(axes.flatten()[class_idx])
        class_shap = shap_values[:, :, class_idx]
        shap.summary_plot(
            class_shap, X_sample,
            max_display=max_display, show=False, plot_size=None,
            cmap="coolwarm",
            color_bar=(class_idx % 2 == 1),
        )
        axes.flatten()[class_idx].set_title(
            f"Class {class_idx}: {class_names[class_idx]}", fontsize=10,
        )

    _harmonize_xlims_and_grid(axes, n_top_row=2)
    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    add_panel_labels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=3)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Dependence plot ───────────────────────────────────────────────────────────
# Hand-picked (feature, color-feature) pairs for the 3x3 dependence figure.
DEFAULT_PANELS: List[Tuple[str, str]] = [
    ("Age", "Gender"),
    ("BMI", "Gender"),
    ("Gender", "BMI"),
    ("Snoring", "Witnessed Apneas"),
    ("High Blood Pressure", "BMI"),
    ("Witnessed Apneas", "Gender"),
    ("Snoring", "BMI"),
    ("Age", "BMI"),
    ("Snoring", "Gender"),
]

# Axis limits / tick labels for categorical or bounded continuous features.
FEAT_AXIS: Dict[str, Dict[str, Any]] = {
    "BMI":                 {"xlim": (10, 70)},
    "Gender":              {"xlim": (-0.5, 1.5),
                            "xticks":      [0, 1],
                            "xticklabels": ["Female", "Male"]},
    "High Blood Pressure": {"xlim": (-1.5, 1.5),
                            "xticks":      [-1, 0, 1],
                            "xticklabels": ["Missing", "No", "Yes"]},
    "Snoring":             {"xlim": (-1.5, 4.5),
                            "xticks":      [-1, 0, 1, 2, 3, 4],
                            "xticklabels": ["Missing", "0", "1", "2", "3", "4"]},
    "Witnessed Apneas":    {"xlim": (-1.5, 4.5),
                            "xticks":      [-1, 0, 1, 2, 3, 4],
                            "xticklabels": ["Missing", "0", "1", "2", "3", "4"]},
}

CATEGORICAL_CBAR: Dict[str, Dict[str, List[Any]]] = {
    "Gender":              {"ticks":  [0, 1],         "labels": ["Female", "Male"]},
    "Witnessed Apneas":    {"ticks":  [-1, 0, 1, 2, 3, 4],
                            "labels": ["Miss.", "0", "1", "2", "3", "4"]},
    "Snoring":             {"ticks":  [-1, 0, 1, 2, 3, 4],
                            "labels": ["Miss.", "0", "1", "2", "3", "4"]},
    "High Blood Pressure": {"ticks":  [-1, 0, 1],     "labels": ["Miss.", "No", "Yes"]},
}

_JITTER_COLS = {"Gender", "High Blood Pressure", "Snoring", "Witnessed Apneas"}
_NAN_FILL_COLS = {"Snoring", "Witnessed Apneas", "High Blood Pressure"}


def _prepare_jittered_X(
    X_display: pd.DataFrame, random_state: int = 42,
) -> pd.DataFrame:
    """Fill NaNs with -1 and add small jitter to discrete columns.

    Without jitter, dependence-plot points stack on top of each other
    and the density structure is invisible.
    """
    rng = np.random.default_rng(random_state)
    X = X_display.copy()
    for col in _NAN_FILL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(-1)
    for col in _JITTER_COLS:
        if col in X.columns:
            X[col] = X[col] + rng.uniform(-0.15, 0.15, size=len(X))
    return X


def plot_shap_dependence_3x3(
    shap_values: np.ndarray,
    X_display: pd.DataFrame,
    save_path: Path,
    target_desc: str,
    panels: List[Tuple[str, str]] | None = None,
    random_state: int = 42,
) -> None:
    """3×3 SHAP dependence plot with hand-picked clinical interactions."""
    panels = panels or DEFAULT_PANELS
    n = len(X_display)

    X_jittered = _prepare_jittered_X(X_display, random_state)
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))

    for idx, (feat, color_feat) in enumerate(panels):
        ax = axes[idx // 3, idx % 3]
        color_idx = list(X_jittered.columns).index(color_feat)
        axes_before = set(fig.axes)

        shap.dependence_plot(
            feat, shap_values, X_jittered,
            interaction_index=color_idx, ax=ax,
            show=False, alpha=0.3, dot_size=5,
        )

        # Re-style the colorbar that shap.dependence_plot just spawned
        new_axes = set(fig.axes) - axes_before
        if color_feat in CATEGORICAL_CBAR and new_axes:
            cbar_ax = new_axes.pop()
            fix = CATEGORICAL_CBAR[color_feat]
            cbar_ax.set_yticks(fix["ticks"])
            cbar_ax.set_yticklabels(fix["labels"], fontsize=8)
            cbar_ax.set_ylabel(color_feat, fontweight="bold", fontsize=10)

        # x-axis customisation
        if feat in FEAT_AXIS:
            cfg = FEAT_AXIS[feat]
            if "xlim" in cfg:
                ax.set_xlim(cfg["xlim"])
            if "xticks" in cfg:
                ax.set_xticks(cfg["xticks"])
            if "xticklabels" in cfg:
                ax.set_xticklabels(cfg["xticklabels"])

        ax.set_title(f"{feat} × {color_feat}", fontsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontweight="bold", fontsize=10)
        ax.set_ylabel(f"SHAP({feat})",  fontweight="bold", fontsize=10)

    fig.suptitle(
        f"SHAP Dependence Plots — Feature Interactions\n"
        f"{target_desc} (Test Set, n = {n:,})",
        fontsize=12, fontweight="bold",
    )
    add_panel_labels(axes)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3, w_pad=2)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
