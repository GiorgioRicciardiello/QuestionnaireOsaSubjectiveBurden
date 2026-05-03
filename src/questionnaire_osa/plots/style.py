"""Shared matplotlib style and panel-labelling helpers."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


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


def apply_publication_style() -> None:
    """Apply the project-wide rcParams (call once at the start of a script)."""
    plt.rcParams.update(_PUBLICATION_RC)


def add_panel_labels(
    axes: Iterable[plt.Axes] | np.ndarray,
    start: str = "a",
    x: float = 0.02,
    y: float = 0.98,
    fontsize: int = 12,
) -> None:
    """Annotate an axes grid with ``(a), (b), (c), …`` labels."""
    flat = np.array(axes).flatten()
    for i, ax in enumerate(flat):
        label = f"({chr(ord(start) + i)})"
        ax.text(
            x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="top", ha="left",
        )
