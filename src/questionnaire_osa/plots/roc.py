"""ROC curves with embedded per-feature-set metric tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from .style import add_panel_labels


# ── Display labels ────────────────────────────────────────────────────────────
_FS_LINE_LABELS = {
    "dem_only":     "Demographics Only",
    "no_dem":       "Questionnaire Only",
    "all_features": "All Features",
}
_FS_LINESTYLES = {"dem_only": "--", "no_dem": "-.", "all_features": "-"}
_FS_TABLE_LABELS = {
    "all_features": "All Features",
    "no_dem":       "Questionnaire",
    "dem_only":     "Demographics",
}


def _add_metrics_table(
    ax: plt.Axes,
    target: str,
    feature_sets: Dict[str, Any],
    fs_order: Sequence[str],
    fs_colors: Dict[str, str],
    scale: float,
) -> None:
    """Embed a small AUC / Sensitivity / Specificity table inside *ax*."""
    col_labels = ["AUC", "Se", "Sp"]
    cell_text = []
    row_colors = []
    for fs_name in fs_order:
        r = feature_sets[fs_name]["all_results"][target]
        cell_text.append([
            f"{r['auroc']:.3f}",
            f"{r['recall']:.3f}",
            f"{r['specificity']:.3f}",
        ])
        row_colors.append(fs_colors[fs_name])

    table = ax.table(
        cellText=cell_text,
        rowLabels=[_FS_TABLE_LABELS[fs] for fs in fs_order],
        rowColours=row_colors,
        colLabels=col_labels,
        bbox=[0.66, 0.02, 0.32, 0.28],
        cellLoc="right",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6 * scale)
    table.set_zorder(5)

    n_rows = len(fs_order)
    n_cols = len(col_labels)

    # Pass 1 — uniform hairlines, snug widths, opaque white data cells
    for (row, col), cell in table.get_celld().items():
        cell.set_width(0.11 if col == -1 else 0.07)
        cell.set_linewidth(0.4)
        cell.set_edgecolor("#aaaaaa")
        if col >= 0 and row > 0:
            cell.set_facecolor("white")

    # Pass 2 — row labels: white bold text, left-aligned
    for i in range(n_rows):
        cell = table[(i + 1, -1)]
        cell.set_text_props(color="white", fontweight="bold")
        cell.get_text().set_ha("left")

    # Pass 3 — column header styling
    for j in range(n_cols):
        cell = table[(0, j)]
        cell.set_facecolor("#e8e8e8")
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#888888")
        cell.set_linewidth(0.5)


def plot_roc_curves(
    feature_sets: Dict[str, Any],
    binary_targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    fs_order: Sequence[str],
    fs_colors: Dict[str, str],
    save_path: Path,
    scale: float = 1.0,
    cell_size: tuple[float, float] = (5.0, 4.0),
) -> None:
    """2×3 ROC grid (one panel per binary target)."""
    n_rows, n_cols = 2, 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size[0], n_rows * cell_size[1]),
    )

    for idx, target in enumerate(binary_targets):
        ax = axes.flatten()[idx]

        for fs_name in fs_order:
            preds = feature_sets[fs_name]["all_predictions"][target]
            fpr, tpr, _ = roc_curve(preds["y_true"], preds["y_proba"])
            ax.plot(
                fpr, tpr,
                color=fs_colors[fs_name], lw=2,
                linestyle=_FS_LINESTYLES[fs_name],
                label=_FS_LINE_LABELS[fs_name],
            )
        ax.plot([0, 1], [0, 1], "k--", lw=1)

        desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
        n = len(feature_sets["all_features"]["all_test_sets"][target]["y_test"])
        ax.set_title(f"{desc}\n(n = {n:,})", fontsize=9 * scale, fontweight="bold")
        ax.set_xlabel("1 − Specificity", fontsize=8 * scale)
        ax.set_ylabel("Sensitivity",     fontsize=8 * scale)
        ax.tick_params(labelsize=7 * scale)
        ax.legend(loc="upper left", fontsize=7 * scale)
        ax.grid(alpha=0.3)
        _add_metrics_table(ax, target, feature_sets, fs_order, fs_colors, scale)

    plt.suptitle(
        "ROC Curves on Test Set — Threshold = 0.50",
        fontsize=11 * scale, fontweight="bold",
    )
    add_panel_labels(axes, x=-0.08, y=1.08)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
