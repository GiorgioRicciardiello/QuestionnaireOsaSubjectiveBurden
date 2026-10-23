"""SHAP dependence figures: a 2×2 main-text and a 2×3 supplementary grid.

Pairs are chosen by mean |SHAP interaction| (see ``interactions.py``): the
strongest, non-redundant pairs go to the main text, weaker symptom/comorbidity
pairs to the supplement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..plots.style import SHAP_CMAP, add_panel_labels
from .annotate import FIG_W, PT_PANEL, PT_SUPTITLE, PT_TITLE

Panel = Tuple[str, str]

# (feature on x-axis, feature used for colour)
MAIN_PANELS: List[Panel] = [
    ("Age", "BMI"),
    ("Age", "Gender"),
    ("BMI", "Gender"),
    ("Age", "Snoring"),
]
SUPPLEMENT_PANELS: List[Panel] = [
    ("Snoring", "BMI"),
    ("Snoring", "Witnessed Apneas"),
    ("Snoring", "Gender"),
    ("Witnessed Apneas", "BMI"),
    ("Age", "High Blood Pressure"),
    ("BMI", "High Blood Pressure"),
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

TREND_COLOR = "#3B3B4F"
N_TREND_BINS = 20
LEGEND_NOTE = (
    "Dashed line: SHAP = 0.  Solid line: mean SHAP in 20 equal-count bins.  "
    "Bars: median SHAP per category, with n.\n"
    "Dotted vertical line: where mean (or category median) SHAP turns positive, i.e. risk-raising."
)


def _fill_missing(X_display: pd.DataFrame) -> pd.DataFrame:
    """Code missing questionnaire answers as -1 (own category)."""
    X = X_display.copy()
    for col in _NAN_FILL_COLS & set(X.columns):
        X[col] = X[col].fillna(-1)
    return X


def _jitter(X: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Add small jitter to discrete columns so points do not stack."""
    rng = np.random.default_rng(random_state)
    out = X.copy()
    for col in _JITTER_COLS & set(out.columns):
        out[col] = out[col] + rng.uniform(-0.15, 0.15, size=len(out))
    return out


def _style_colorbar(cbar_ax: plt.Axes, color_feat: str) -> None:
    # shap's point alpha leaks into the colorbar; restore full colour
    for artist in (*cbar_ax.collections, *cbar_ax.images):
        artist.set_alpha(1)
    cbar_ax.set_ylabel(color_feat, fontweight="bold", fontsize=6, labelpad=4)
    cbar_ax.yaxis.set_label_position("right")
    cbar_ax.tick_params(labelsize=5, length=2)
    if color_feat in CATEGORICAL_CBAR:
        fix = CATEGORICAL_CBAR[color_feat]
        lo, hi = cbar_ax.get_ylim()
        keep = [(t, l) for t, l in zip(fix["ticks"], fix["labels"]) if lo <= t <= hi]
        cbar_ax.set_yticks([t for t, _ in keep])
        cbar_ax.set_yticklabels([l for _, l in keep], fontsize=5)


def _zero_crossing(x: np.ndarray, y: np.ndarray) -> float | None:
    """x where ``y`` first goes from negative to non-negative (linear interpolation)."""
    for i in range(len(x) - 1):
        if y[i] < 0 <= y[i + 1]:
            return float(x[i] + (x[i + 1] - x[i]) * (-y[i]) / (y[i + 1] - y[i]))
    return None


def _mark_shift(ax: plt.Axes, x_line: float | None, label: str) -> None:
    """Vertical line at the point where the feature starts raising risk."""
    if x_line is None:
        return
    ax.axvline(x_line, color=TREND_COLOR, lw=0.8, ls=":", zorder=3)
    ax.text(x_line, 0.96, f" {label}", transform=ax.get_xaxis_transform(),
            ha="left", va="top", fontsize=5.5, fontweight="bold", color=TREND_COLOR)


def _overlay_trend(ax: plt.Axes, x: np.ndarray, y: np.ndarray, categorical: bool) -> None:
    """Zero line, binned mean / category medians, and the zero-crossing shift."""
    ax.axhline(0, color="#777777", lw=0.6, ls="--", zorder=1)
    if categorical:
        trans = ax.get_xaxis_transform()
        cats = np.unique(x)
        medians = []
        for cat in cats:
            sel = x == cat
            medians.append(np.median(y[sel]))
            ax.hlines(medians[-1], cat - 0.3, cat + 0.3,
                      color=TREND_COLOR, lw=1.6, zorder=4)
            ax.text(cat, 0.02, f"n =\n{sel.sum():,}", transform=trans,
                    ha="center", va="bottom", fontsize=4.3, color="#555555",
                    linespacing=0.9)
        # Missing (-1) is not on the ordinal scale; the shift is the first
        # answered level whose median SHAP is positive, drawn between levels.
        positive = [c for c, m in zip(cats, medians) if c >= 0 and m > 0]
        if positive:
            _mark_shift(ax, positive[0] - 0.5, f"≥ {positive[0]:.0f}")
        return
    order = np.argsort(x)
    chunks = np.array_split(order, N_TREND_BINS)
    bx = np.array([x[c].mean() for c in chunks])
    by = np.array([y[c].mean() for c in chunks])
    ax.plot(bx, by, color=TREND_COLOR, lw=1.4, zorder=4)
    x_cross = _zero_crossing(bx, by)
    if x_cross is not None:
        _mark_shift(ax, x_cross, f"{x_cross:.1f}")


def _draw_panel(
    fig: plt.Figure, ax: plt.Axes, feat: str, color_feat: str,
    shap_values: np.ndarray, X_plot: pd.DataFrame, X_true: pd.DataFrame,
) -> None:
    axes_before = set(fig.axes)
    shap.dependence_plot(
        feat, shap_values, X_plot,
        interaction_index=list(X_plot.columns).index(color_feat), ax=ax,
        show=False, alpha=0.45, dot_size=1.5, cmap=SHAP_CMAP,
    )
    for cbar_ax in set(fig.axes) - axes_before:
        _style_colorbar(cbar_ax, color_feat)

    col = list(X_plot.columns).index(feat)
    _overlay_trend(ax, X_true[feat].to_numpy(float), shap_values[:, col],
                   categorical=feat in _JITTER_COLS)

    cfg = FEAT_AXIS.get(feat, {})
    if "xlim" in cfg:
        ax.set_xlim(cfg["xlim"])
    if "xticks" in cfg:
        ax.set_xticks(cfg["xticks"])
    if "xticklabels" in cfg:
        ax.set_xticklabels(cfg["xticklabels"])

    ax.set_title(f"{feat} × {color_feat}", fontsize=PT_TITLE)
    ax.set_xlabel(feat, fontweight="bold", fontsize=6)
    ax.set_ylabel(f"SHAP({feat})", fontweight="bold", fontsize=6)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)


def plot_shap_dependence_grid(
    shap_values: np.ndarray,
    X_display: pd.DataFrame,
    save_path: Path,
    title: str,
    panels: List[Panel],
    n_rows: int,
    n_cols: int,
    fig_h: float,
    random_state: int = 42,
) -> None:
    """Dependence grid for ``panels`` with zero line, trend and category medians."""
    X_true = _fill_missing(X_display)
    X_plot = _jitter(X_true, random_state)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_W, fig_h), squeeze=False)

    for ax, (feat, color_feat) in zip(axes.flatten(), panels):
        _draw_panel(fig, ax, feat, color_feat, shap_values, X_plot, X_true)

    fig.suptitle(f"{title} (Test Set, n = {len(X_display):,})",
                 fontsize=PT_SUPTITLE, fontweight="bold", y=0.995)
    add_panel_labels(axes, fontsize=PT_PANEL)
    fig.text(0.5, 0.003, LEGEND_NOTE, ha="center", va="bottom",
             fontsize=5.5, color="#555555")
    plt.tight_layout(rect=[0, 0.04, 1, 1.0], h_pad=1.0, w_pad=0.8)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence_main(
    shap_values: np.ndarray, X_display: pd.DataFrame, save_path: Path,
    target_desc: str, random_state: int = 42,
) -> None:
    """Main-text figure: the four strongest, non-redundant interactions."""
    plot_shap_dependence_grid(
        shap_values, X_display, save_path,
        f"SHAP Dependence — Principal Interactions\n{target_desc}",
        MAIN_PANELS, 2, 2, 4.9, random_state,
    )


def plot_shap_dependence_supplement(
    shap_values: np.ndarray, X_display: pd.DataFrame, save_path: Path,
    target_desc: str, random_state: int = 42,
) -> None:
    """Supplementary figure: weaker symptom and comorbidity interactions."""
    plot_shap_dependence_grid(
        shap_values, X_display, save_path,
        f"SHAP Dependence — Additional Interactions\n{target_desc}",
        SUPPLEMENT_PANELS, 2, 3, 4.6, random_state,
    )
