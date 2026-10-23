"""Shared matplotlib style and panel-labelling helpers."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


_PUBLICATION_RC = {
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "sans-serif",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.titlesize":  11,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "savefig.dpi":       300,
}


# Pastel, blue-free diverging map for SHAP feature-value colouring:
# low = green, mid = butter, high = coral-rose; every stop is
# saturated enough to read on white (no near-white midpoint) (matches Figures 1-2).
SHAP_CMAP = LinearSegmentedColormap.from_list(
    "shap_pastel", ["#5FB894", "#B5D98F", "#F6C976", "#EE9A7A", "#D9606F"],
)


def apply_publication_style() -> None:
    """Apply the project-wide rcParams (call once at the start of a script)."""
    plt.rcParams.update(_PUBLICATION_RC)


def add_panel_labels(
    axes: Iterable[plt.Axes] | np.ndarray,
    start: str = "a",
    x: float = 0.02,
    y: float = 0.98,
    fontsize: int = 12,
    va: str = "top",
) -> None:
    """Annotate an axes grid with ``(a), (b), (c), …`` labels."""
    flat = np.array(axes).flatten()
    for i, ax in enumerate(flat):
        label = f"({chr(ord(start) + i)})"
        ax.text(
            x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va=va, ha="left",
        )
