"""Manuscript Figure 2 — TabPFN confusion matrices, eight panels, no empty space.

Row 1 is AHI (≥5, ≥15, ≥30, four-class); row 2 is RDI.  Every panel has the
same outer size so the grid is gap-free; the four-class matrices simply carry
smaller cells.  Colour encodes severity: each endpoint group has its own pastel
family that deepens with the threshold, and cell intensity within a panel is
the row-normalised percentage.  Counts are the mean of the five CV folds
(``counts="mean"``) or their sum (``counts="sum"``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

TARGETS = (
    ("ahi_5", "ahi_15", "ahi_30", "ahi_4class"),
    ("rdi_5", "rdi_15", "rdi_30", "rdi_4class"),
)
INDEX_LABELS = ("AHI", "RDI")
THRESHOLDS = (5, 15, 30)
BINARY_LABELS = ("No OSA", "OSA")
CLASS_LABELS = ("None", "Mild", "Moderate", "Severe")

INK = "#222222"
SUBTLE = "#555555"

# ── Palettes: near-white → pastel anchor, deepening with severity ────────────
PALETTES: Mapping[str, tuple[str, ...]] = {
    "ahi": ("#F6D88C", "#F4A988", "#E2707A"),     # butter → apricot → coral-rose
    "rdi": ("#BFE3C8", "#86C69E", "#4FA57A"),     # mint → sage → pastel green
}
CLASS_CELL_ANCHOR = "#A98AD1"                     # darkest cell colour, 4-class
NEAR_WHITE = "#FFFFFF"

# ── Geometry (inches) ────────────────────────────────────────────────────────
FIG_W = 7.08                                      # 180 mm
LEFT, RIGHT = 0.60, 0.04
HGAPS = (0.07, 0.07, 0.17)            # the last gap holds the 4-class y ticks
VGAP = 0.30
BOTTOM, TOP = 0.34, 0.22
PANEL = (FIG_W - LEFT - RIGHT - sum(HGAPS)) / 4
COL_X = tuple(LEFT + col * PANEL + sum(HGAPS[:col]) for col in range(4))
FIG_H = BOTTOM + TOP + 2 * PANEL + VGAP

# ── Type (pt) ────────────────────────────────────────────────────────────────
PT_TITLE, PT_LABEL, PT_TICK = 6.8, 6.5, 6.0
PT_COUNT_BIN, PT_PCT_BIN = 8.5, 6.8
PT_COUNT_CLS, PT_PCT_CLS = 6.2, 5.2
PT_TICK_CLS = 5.4
CELL_GAP = 1.4                                    # white gap between cells (pt)

RC = {
    "font.family":     "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       PT_TICK,
    "text.color":      INK,
    "axes.linewidth":  0.0,
    "xtick.color":     INK,
    "ytick.color":     INK,
    "xtick.major.size": 0.0,
    "ytick.major.size": 0.0,
    "xtick.major.pad": 1.6,
    "ytick.major.pad": 1.6,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "svg.fonttype":    "none",
}


# ── Pure data helpers ────────────────────────────────────────────────────────
def aggregate_counts(folds: np.ndarray, mode: str = "mean") -> np.ndarray:
    """Collapse the per-fold (n_folds, k, k) stack into one integer matrix."""
    folds = np.asarray(folds)
    if mode == "mean":
        return np.rint(folds.mean(axis=0)).astype(int)
    if mode == "sum":
        return folds.sum(axis=0).astype(int)
    raise ValueError(f"counts must be 'mean' or 'sum', got {mode!r}")


def sample_size(folds: np.ndarray, mode: str = "mean") -> int:
    """Patients per panel, taken from the raw fold totals (not the rounded cells)."""
    total = np.asarray(folds).sum()
    return int(round(total / len(folds))) if mode == "mean" else int(total)


def row_percent(cm: np.ndarray) -> np.ndarray:
    """Row-normalised percentages; an all-zero row stays at 0."""
    totals = cm.sum(axis=1, keepdims=True).astype(float)
    return np.divide(cm, totals, out=np.zeros(cm.shape, float), where=totals > 0) * 100


def _cmap(anchor: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("pastel", [NEAR_WHITE, anchor])


def _panel_anchor(row: int, col: int) -> str:
    if col == 3:
        return CLASS_CELL_ANCHOR
    return PALETTES["ahi" if row == 0 else "rdi"][col]


# ── Drawing ──────────────────────────────────────────────────────────────────
def _draw_matrix(ax: plt.Axes, cm: np.ndarray, anchor: str,
                 labels: Sequence[str], four_class: bool,
                 show_x: bool, show_y: bool) -> None:
    pct = row_percent(cm)
    k = len(labels)
    ax.imshow(pct, cmap=_cmap(anchor), vmin=0, vmax=100, aspect="equal",
              interpolation="nearest")
    for edge in np.arange(-0.5, k, 1.0):
        ax.axhline(edge, color="white", lw=CELL_GAP)
        ax.axvline(edge, color="white", lw=CELL_GAP)

    pt_count, pt_pct = (PT_COUNT_CLS, PT_PCT_CLS) if four_class else (PT_COUNT_BIN, PT_PCT_BIN)
    shift = 0.13 if four_class else 0.11
    for i in range(k):
        for j in range(k):
            ax.text(j, i - shift, f"{cm[i, j]:,}", ha="center", va="center",
                    fontsize=pt_count, fontweight="bold", color=INK)
            ax.text(j, i + shift + 0.03, f"{pct[i, j]:.1f}%", ha="center",
                    va="center", fontsize=pt_pct, color=SUBTLE if not four_class else INK)

    tick_pt = PT_TICK_CLS if four_class else PT_TICK
    if show_x:
        ax.set_xticks(range(k), labels, fontsize=tick_pt)
    else:
        ax.set_xticks([])
    if show_y:
        ax.set_yticks(range(k), labels, fontsize=tick_pt, rotation=90, va="center")
    else:
        ax.set_yticks([])
    ax.set_xlim(-0.5, k - 0.5)
    ax.set_ylim(k - 0.5, -0.5)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _title(ax: plt.Axes, letter: str, text: str) -> None:
    ax.text(-0.02, 1.035, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, fontweight="bold")
    ax.text(0.5, 1.035, text, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=PT_TITLE)


def _panel_title(row: int, col: int, n: int) -> str:
    what = f"{INDEX_LABELS[row]} ≥ {THRESHOLDS[col]}" if col < 3 else f"{INDEX_LABELS[row]} 4-class"
    return f"{what} · n = {n:,}"


def _save(fig: plt.Figure, stem: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        extra = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "tiff" else {}
        fig.savefig(path, dpi=dpi, facecolor="white", **extra)
        written.append(path)
    return written


def plot_confusion_figure(
    folds_by_target: Mapping[str, np.ndarray],
    save_stem: Path,
    counts: str = "mean",
    formats: Sequence[str] = ("pdf", "svg", "png", "tiff"),
    dpi: int = 600,
) -> list[Path]:
    """Draw Figure 2 from per-fold confusion matrices and write each format."""
    expected = [t for row in TARGETS for t in row]
    missing = [t for t in expected if t not in folds_by_target]
    if missing:
        raise KeyError(f"Confusion matrices missing for targets: {missing}")

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        for row, targets in enumerate(TARGETS):
            for col, target in enumerate(targets):
                cm = aggregate_counts(folds_by_target[target], counts)
                x0 = COL_X[col]
                y0 = BOTTOM + (1 - row) * (PANEL + VGAP)
                ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, PANEL / FIG_W, PANEL / FIG_H])
                four = col == 3
                _draw_matrix(ax, cm, _panel_anchor(row, col),
                             CLASS_LABELS if four else BINARY_LABELS, four,
                             show_x=(row == 1), show_y=(col in (0, 3)))
                _title(ax, chr(ord("a") + row * 4 + col), _panel_title(row, col, sample_size(folds_by_target[target], counts)))

            y_mid = (BOTTOM + (1 - row) * (PANEL + VGAP) + PANEL / 2) / FIG_H
            fig.text(0.05 / FIG_W, y_mid, INDEX_LABELS[row], rotation=90,
                     ha="left", va="center", fontsize=8, fontweight="bold")
            fig.text(0.26 / FIG_W, y_mid, "Actual", rotation=90, ha="left",
                     va="center", fontsize=PT_LABEL, color=SUBTLE)

        grid_cx = (LEFT + (4 * PANEL + sum(HGAPS)) / 2) / FIG_W
        fig.text(grid_cx, 0.02 / FIG_H, "Predicted", ha="center", va="bottom",
                 fontsize=PT_LABEL, color=SUBTLE)
        written = _save(fig, save_stem, formats, dpi)
        plt.close(fig)
    return written
