"""Manuscript Figure 3: binary beeswarms for several feature sets in one grid."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.ticker import MaxNLocator

from ..plots.style import SHAP_CMAP, add_panel_labels
from .annotate import (
    FIG_W, PT_PANEL, PT_SUPTITLE, PT_TITLE, add_note, finish_beeswarm,
    label_mean_abs,
)
from .compute import exclude_feature_from_shap

ModelRow = Tuple[str, Dict[str, Dict[str, Any]]]


def plot_shap_beeswarm_models(
    rows: Sequence[ModelRow],
    targets: Sequence[str],
    targets_config: Dict[str, Dict[str, Any]],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
    fig_h: float = 7.4,
) -> None:
    """One row per feature set (label, SHAP data), one column per target.

    The x-axis is shared within a row only: the questionnaire-only model
    attributes ~10x less than the all-features model.
    """
    n_rows, n_cols = len(rows), len(targets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, fig_h), squeeze=False)

    for r, (_, shap_data) in enumerate(rows):
        for c, target in enumerate(targets):
            ax = axes[r, c]
            plt.sca(ax)
            d = shap_data[target]
            sv, xs = exclude_feature_from_shap(d["shap_values"], d["X_sample"])
            shap.summary_plot(
                sv, xs, max_display=max_display, show=False, plot_size=None,
                cmap=SHAP_CMAP, color_bar=(c == n_cols - 1),
            )
            ax.set_title(targets_config[target]["desc"].replace(" @ ", " ≥ "))
            label_mean_abs(ax, sv, xs)

        limits = [ax.get_xlim() for ax in axes[r]]
        lo, hi = min(l for l, _ in limits), max(h for _, h in limits)
        for c, ax in enumerate(axes[r]):
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            ax.grid(axis="x", alpha=0.7)
            ax.set_xlabel("SHAP Value" if r == n_rows - 1 else "")

    finish_beeswarm(fig, axes)
    add_panel_labels(axes, x=-0.02, y=1.0, fontsize=PT_PANEL, va="bottom")
    fig.suptitle(suptitle, fontsize=PT_SUPTITLE, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94], h_pad=3.2, w_pad=0.6)

    # Row headers centred above each row, between the rows' panel titles.
    for r, (label, _) in enumerate(rows):
        left, right = axes[r, 0].get_position(), axes[r, -1].get_position()
        y = left.y1 + 0.038
        fig.text((left.x0 + right.x1) / 2, y, label, ha="center", va="bottom",
                 fontsize=PT_SUPTITLE - 0.5, fontweight="bold")
    add_note(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_strata_combined(
    blocks: Sequence[Tuple[str, Dict[str, Dict[str, Any]]]],
    suptitle: str,
    save_path: Path,
    max_display: int = 10,
    fig_h: float = 9.4,
) -> None:
    """Stacked 2×2 strata grids (e.g. age over BMI) for one target.

    ``blocks`` holds (header, groups) with ``groups`` as returned by
    :func:`shap_by_strata`; the x-axis is shared across all panels.
    """
    from .strata import display_group_label

    n_rows = 2 * len(blocks)
    fig, axes = plt.subplots(n_rows, 2, figsize=(FIG_W, fig_h), squeeze=False)

    for b, (_, groups) in enumerate(blocks):
        for g, (name, data) in enumerate(groups.items()):
            ax = axes[2 * b + g // 2, g % 2]
            plt.sca(ax)
            sv, xs = exclude_feature_from_shap(data["shap_values"], data["X_subset"])
            shap.summary_plot(
                sv, xs, max_display=max_display, show=False, plot_size=None,
                cmap=SHAP_CMAP, color_bar=(g % 2 == 1),
            )
            ax.set_title(f"{display_group_label(name)} (n = {data['n_samples']:,})")
            label_mean_abs(ax, sv, xs)

    limits = [ax.get_xlim() for ax in axes.flatten()]
    lo, hi = min(l for l, _ in limits), max(h for _, h in limits)
    for r, row in enumerate(axes):
        for ax in row:
            ax.set_xlim(lo, hi)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
            ax.grid(axis="x", alpha=0.7)
            ax.set_xlabel("SHAP Value" if r == n_rows - 1 else "")

    finish_beeswarm(fig, axes)
    add_panel_labels(axes, x=-0.02, y=1.0, fontsize=PT_PANEL, va="bottom")
    fig.suptitle(suptitle, fontsize=PT_SUPTITLE, fontweight="bold")
    plt.tight_layout(rect=[0, 0.015, 1, 0.965], h_pad=2.6, w_pad=0.6)

    for b, (header, _) in enumerate(blocks):
        top, right = axes[2 * b, 0].get_position(), axes[2 * b, 1].get_position()
        fig.text((top.x0 + right.x1) / 2, top.y1 + 0.03, header, ha="center",
                 va="bottom", fontsize=PT_SUPTITLE - 0.5, fontweight="bold")
    add_note(fig)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
