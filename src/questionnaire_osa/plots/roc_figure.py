"""Manuscript Figure 1 — held-out ROC curves, six panels, nothing else.

2 × 3 grid (rows AHI / RDI, columns ≥5 / ≥15 / ≥30 events/h).  Every panel is
as large as the 180 mm page width allows; the AUC (DeLong 95 % CI),
sensitivity and specificity of the three models sit in a small table in the
empty lower-right corner of the panel.  Se/Sp are evaluated at the plotted operating
point (probability threshold τ = 0.50).  The shaded bands are pointwise 95 % bootstrap CIs of the ROC curve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from ..reporting.reference_run import endpoint_metrics


MODELS = ("all_features", "dem_only", "no_dem")          # strongest first
MODEL_LABEL = {
    "all_features": "All features",
    "dem_only":     "Demographics only",
    "no_dem":       "Questionnaire only",
}
MODEL_COLOR = {
    "all_features": "#2c115f",    # magma 0.20
    "dem_only":     "#b73779",    # magma 0.50
    "no_dem":       "#f8765c",    # magma 0.75
}
INDEX_LABELS = ("AHI", "RDI")
THRESHOLDS = (5, 15, 30)

INK = "#141414"
SUBTLE = "#6e6e6e"

# ── Geometry (inches) ─────────────────────────────────────────────────────────
PANEL = 1.55
LEFT, RIGHT = 0.50, 0.06
HGAP, VGAP = 0.12, 0.30
BOTTOM, TOP = 0.46, 0.40
FIG_W = LEFT + 3 * PANEL + 2 * HGAP + RIGHT    # ≈ 6.2 in (157 mm)

# ── Type (pt) ─────────────────────────────────────────────────────────────────
PT_TITLE = 7.0
PT_LABEL = 6.5
PT_TICK = 6.0
PT_CELL = 5.6

LW_ROC = 1.1
TAU = 0.50                                     # operating-point threshold
OP_MARK = dict(marker="o", ms=4.0, mew=0.7, mec="white", ls="none")
BAND_ALPHA = 0.35

N_BOOT = 1000
SEED = 20261020
FPR_GRID = np.linspace(0.0, 1.0, 201)

RC = {
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         PT_TICK,
    "text.color":        INK,
    "axes.linewidth":    0.5,
    "axes.edgecolor":    INK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.labelsize":   PT_TICK,
    "ytick.labelsize":   PT_TICK,
    "xtick.color":       INK,
    "ytick.color":       INK,
    "xtick.major.size":  2.0,
    "ytick.major.size":  2.0,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.pad":   1.8,
    "ytick.major.pad":   1.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
}


class Curve(NamedTuple):
    model: str
    fpr: np.ndarray
    tpr: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    auc: float
    auc_lo: float
    auc_hi: float
    se: float
    sp: float
    n: int
    n_pos: int


# ── Statistics ────────────────────────────────────────────────────────────────
def _tpr_on_grid(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    fpr, tpr, _ = roc_curve(y, p)
    out = np.interp(FPR_GRID, fpr, tpr)
    out[0] = 0.0
    return out


def _bands(y: np.ndarray, probas: Sequence[np.ndarray],
           rng: np.random.Generator) -> np.ndarray:
    """(model, 2, grid) of 2.5 / 97.5 % TPR quantiles; resamples are paired
    across models and stratified by class."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    draws = np.empty((len(probas), N_BOOT, FPR_GRID.size))
    for b in range(N_BOOT):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        for m, p in enumerate(probas):
            draws[m, b] = _tpr_on_grid(y[idx], p[idx])
    return np.moveaxis(np.percentile(draws, [2.5, 97.5], axis=1), 0, 1)


def endpoint_curves(feature_sets: Dict[str, Any], target: str,
                    rng: np.random.Generator) -> list[Curve]:
    summary = endpoint_metrics(feature_sets, target)
    stats = summary["fs"]
    preds = [feature_sets[m]["all_predictions"][target] for m in MODELS]
    y = np.asarray(preds[0]["y_true"]).astype(int)
    probas = [np.asarray(p["y_proba"], dtype=float) for p in preds]
    bands = _bands(y, probas, rng)

    curves = []
    for model, proba, (lo, hi) in zip(MODELS, probas, bands):
        fpr, tpr, _ = roc_curve(y, proba)
        s = stats[model]
        curves.append(Curve(
            model=model, fpr=fpr, tpr=tpr, lo=lo, hi=hi,
            auc=float(s["auroc"]), auc_lo=float(s["auroc_ci"][0]),
            auc_hi=float(s["auroc_ci"][1]),
            se=float((proba[y == 1] >= TAU).mean()),
            sp=float((proba[y == 0] < TAU).mean()),
            n=int(summary["n"]), n_pos=int(summary["n_pos"]),
        ))
    return curves


# ── Drawing ───────────────────────────────────────────────────────────────────
def _style_axes(ax: plt.Axes, show_x: bool, show_y: bool) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    labels = ["0", "", "0.5", "", "1"]
    ax.set_xticklabels(labels if show_x else [])
    ax.set_yticklabels(labels if show_y else [])
    ax.set_aspect("equal", adjustable="box")


def _draw_curves(ax: plt.Axes, curves: Sequence[Curve]) -> None:
    ax.plot([0, 1], [0, 1], color="#c4c4c4", lw=0.5, zorder=1)
    for z, c in enumerate(reversed(curves)):                 # strongest on top
        col = MODEL_COLOR[c.model]
        ax.fill_between(FPR_GRID, c.lo, c.hi, facecolor=col, alpha=BAND_ALPHA,
                        edgecolor="none", lw=0, zorder=2 + z)
        ax.plot(c.fpr, c.tpr, color=col, lw=LW_ROC, solid_joinstyle="round",
                zorder=6 + z)
    for c in curves:
        ax.plot(1 - c.sp, c.se, mfc=MODEL_COLOR[c.model], zorder=12, **OP_MARK)


def _pct(x: float, unit: str = "%") -> str:
    """0.7643 → '76%'."""
    return f"{100 * x:.0f}{unit}"


def _metric_table(ax: plt.Axes, curves: Sequence[Curve]) -> None:
    """AUC (95% CI) · Se · Sp per model, lower-right of the panel."""
    row_h, y0 = 0.125 / PANEL, 0.025                  # 0.125 in per row
    swatch = (1 - 1.22 / PANEL, 1 - 1.15 / PANEL)
    x_auc, x_se, x_sp = (1 - d / PANEL for d in (0.85, 0.35, 0.11))   # column centres
    n = len(curves)

    head_y = y0 + n * row_h + row_h * 0.45
    head = dict(transform=ax.transAxes, fontsize=PT_CELL, color=SUBTLE,
                ha="center", va="center")
    ax.text(x_auc, head_y, "AUC (95% CI)", **head)
    ax.text(x_se, head_y, "Se", **head)
    ax.text(x_sp, head_y, "Sp", **head)

    cell = dict(transform=ax.transAxes, fontsize=PT_CELL, ha="center", va="center")
    for i, c in enumerate(curves):
        y = y0 + (n - 1 - i + 0.5) * row_h
        ax.plot(swatch, [y, y], transform=ax.transAxes, color=MODEL_COLOR[c.model],
                lw=2.0, solid_capstyle="butt", clip_on=False)
        ax.text(x_auc, y, f"{_pct(c.auc)} ({_pct(c.auc_lo, '')}–{_pct(c.auc_hi, '')})",
                color=INK, **cell)
        ax.text(x_se, y, _pct(c.se), color=INK, **cell)
        ax.text(x_sp, y, _pct(c.sp), color=INK, **cell)


def _sample_size(ax: plt.Axes, c: Curve) -> None:
    ax.text(0.04, 0.965, f"n = {c.n:,}\nevents = {c.n_pos:,}", transform=ax.transAxes,
            fontsize=PT_CELL, color=SUBTLE, ha="left", va="top", linespacing=1.3)


def _title(ax: plt.Axes, letter: str, index: str, threshold: int) -> None:
    ax.text(0.0, 1.03, letter, transform=ax.transAxes, fontsize=PT_TITLE + 1,
            fontweight="bold", ha="left", va="bottom")
    ax.text(0.095, 1.03, f"{index} ≥ {threshold} events/h",
            transform=ax.transAxes, fontsize=PT_TITLE, ha="left", va="bottom")


def _legend(fig: plt.Figure, y: float) -> None:
    handles = [Line2D([], [], color=MODEL_COLOR[m], lw=1.8, label=MODEL_LABEL[m])
               for m in MODELS]
    handles.append(Patch(fc="#8c8c8c", alpha=BAND_ALPHA + 0.1, ec="none",
                         label="95% CI"))
    handles.append(Line2D([], [], mfc="#8c8c8c", label=f"τ = {TAU:.2f} (Se, Sp)",
                          **OP_MARK))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y),
               ncol=len(handles), frameon=False, fontsize=PT_LABEL,
               handlelength=1.6, handleheight=0.7, handletextpad=0.5,
               columnspacing=1.8, borderaxespad=0, borderpad=0)


def _save(fig: plt.Figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
        fig.savefig(path, dpi=dpi, facecolor="white", **extra)
        written.append(path)
    return written


def plot_roc_figure(
    feature_sets: Dict[str, Any],
    binary_targets: Sequence[str],
    save_stem: Path,
    formats: Sequence[str] = ("pdf", "svg", "png", "tiff"),
    dpi: int = 600,
) -> list[Path]:
    """Draw Figure 1 and write ``save_stem.<fmt>`` for each format.

    ``binary_targets`` must be ordered AHI ≥5/15/30 then RDI ≥5/15/30.
    """
    if len(binary_targets) != 6:
        raise ValueError(f"Figure 1 expects 6 endpoints, got {len(binary_targets)}")
    missing = [m for m in MODELS if m not in feature_sets]
    if missing:
        raise KeyError(f"Feature sets missing from run: {missing}")

    rng = np.random.default_rng(SEED)
    fig_h = BOTTOM + 2 * PANEL + VGAP + TOP

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, fig_h))
        for i, target in enumerate(binary_targets):
            row, col = divmod(i, 3)
            x0 = LEFT + col * (PANEL + HGAP)
            y0 = BOTTOM + (1 - row) * (PANEL + VGAP)
            ax = fig.add_axes([x0 / FIG_W, y0 / fig_h, PANEL / FIG_W, PANEL / fig_h])
            _style_axes(ax, show_x=(row == 1), show_y=(col == 0))
            curves = endpoint_curves(feature_sets, target, rng)
            _draw_curves(ax, curves)
            _metric_table(ax, curves)
            _sample_size(ax, curves[0])
            _title(ax, chr(ord("a") + i), INDEX_LABELS[row], THRESHOLDS[col])

        grid_cx = (LEFT + (3 * PANEL + 2 * HGAP) / 2) / FIG_W
        fig.text(grid_cx, 0.04 / fig_h, "1 − specificity", ha="center",
                 va="bottom", fontsize=PT_LABEL)
        fig.text(0.05 / FIG_W, (BOTTOM + PANEL + VGAP / 2) / fig_h, "Sensitivity",
                 ha="left", va="center", rotation=90, fontsize=PT_LABEL)
        _legend(fig, y=1 - 0.13 / fig_h)

        written = _save(fig, save_stem, formats, dpi)
        plt.close(fig)
    return written
