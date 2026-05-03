"""Confusion-matrix grids — counts overlaid with row-normalised percentages."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .style import add_panel_labels


def _annotated_heatmap(
    ax: plt.Axes,
    cm: np.ndarray,
    xy_labels: Sequence[str],
    title: str,
    annot_fontsize: float,
) -> None:
    """Draw one heatmap panel: row-normalised % + raw count."""
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    annot = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_pct)
    ])
    sns.heatmap(
        cm_pct, annot=annot, fmt="", cmap="Blues",
        xticklabels=xy_labels, yticklabels=xy_labels, ax=ax,
        cbar=False, vmin=0, vmax=100,
        linewidths=2, linecolor="white",
        annot_kws={"fontsize": annot_fontsize},
    )
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")


def plot_confusion_matrices_binary(
    primary_results: Dict[str, Dict[str, Any]],
    test_sets: Dict[str, Dict[str, Any]],
    binary_targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    save_path: Path,
    scale: float = 1.0,
    cell_size: tuple[float, float] = (4.0, 3.5),
) -> None:
    """2×3 grid of binary confusion matrices for the primary feature set."""
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size[0], n_rows * cell_size[1]),
    )

    for idx, target in enumerate(binary_targets):
        ax = axes.flatten()[idx]
        cm = primary_results[target]["confusion_matrix"]
        desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
        n = len(test_sets[target]["y_test"])
        _annotated_heatmap(
            ax, cm, ["No OSA", "OSA"],
            f"{desc}\n(n = {n:,})", annot_fontsize=10 * scale,
        )

    plt.suptitle(
        "Confusion Matrices — Binary on Test Set (All Features)  τ = 0.50",
        fontsize=12 * scale, fontweight="bold",
    )
    add_panel_labels(axes, x=-0.08, y=1.08)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices_multiclass(
    primary_results: Dict[str, Dict[str, Any]],
    test_sets: Dict[str, Dict[str, Any]],
    multiclass_targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    save_path: Path,
    scale: float = 1.0,
    cell_size: tuple[float, float] = (4.5, 4.0),
) -> None:
    """1×2 grid of 4-class confusion matrices."""
    fig, axes = plt.subplots(
        1, 2, figsize=(2 * cell_size[0], cell_size[1]),
    )

    for idx, target in enumerate(multiclass_targets):
        ax = axes[idx]
        cm = primary_results[target]["confusion_matrix"]
        desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
        n = len(test_sets[target]["y_test"])
        _annotated_heatmap(
            ax, cm, ["None", "Mild", "Moderate", "Severe"],
            f"{desc}\n(n = {n:,})", annot_fontsize=9 * scale,
        )

    plt.suptitle(
        "Confusion Matrices — Four-Class on Test Set (All Features)",
        fontsize=12 * scale, fontweight="bold",
    )
    add_panel_labels(axes, x=-0.08, y=1.08)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
