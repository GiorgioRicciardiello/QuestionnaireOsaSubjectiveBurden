"""Shared annotation helpers for the SHAP beeswarm figures."""

from __future__ import annotations

from typing import Sequence

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MEAN_ABS_NOTE = "Values in brackets: mean |SHAP| (average impact on the model output)"

# Severity bands per class index (same cut-offs for AHI and RDI).
FIG_W = 7.08          # 180 mm journal width (inches)
PT_TICK, PT_TITLE, PT_SUPTITLE, PT_PANEL = 5.5, 6.8, 8.0, 8.0
WRAP_CHARS = 30
DOT_AREA = 3.0            # pt^2; SHAP default (16) is too heavy at 180 mm
MISSING_COLOR = "#C9BFD9"   # pastel lavender-grey replaces SHAP's default grey
_SHAP_GREY = 0.4667         # 0x77/255

CLASS_BANDS = ("< 5", "5 – < 15", "15 – < 30", "≥ 30")
CLASS_FORMAL = ("No OSA", "Mild OSA", "Moderate OSA", "Severe OSA")


def class_title(index_name: str, class_idx: int) -> str:
    """Formal class name with the index cut-off, e.g. ``Mild OSA (AHI 5 – < 15)``."""
    return f"{CLASS_FORMAL[class_idx]} ({index_name} {CLASS_BANDS[class_idx]})"


def label_mean_abs(ax: plt.Axes, shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """Append ``[mean |SHAP|]`` to every y tick label of a beeswarm axis."""
    mean_abs = dict(zip(X.columns, np.abs(shap_values).mean(axis=0)))
    labels = []
    for tick in ax.get_yticklabels():
        name = tick.get_text()
        labels.append(
            textwrap.fill(f"{name} [{mean_abs[name]:.2f}]", WRAP_CHARS)
            if name in mean_abs else name
        )
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(labels)


def add_note(fig: plt.Figure, text: str = MEAN_ABS_NOTE) -> None:
    fig.text(0.5, 0.003, text, ha="center", va="bottom", fontsize=5.5, color="#555555")


def finish_beeswarm(fig: plt.Figure, axes: np.ndarray) -> None:
    """Journal-size fonts, pastel missing-value colour, compact colorbars."""
    flat = list(np.asarray(axes).flatten())
    for ax in flat:
        ax.tick_params(axis="both", labelsize=PT_TICK, length=2, pad=1.5)
        ax.title.set_fontsize(PT_TITLE)
        ax.set_xlabel(ax.get_xlabel(), fontsize=PT_TICK + 0.5, labelpad=1.5)
        for coll in ax.collections:
            coll.set_sizes([DOT_AREA])
            fc = coll.get_facecolor()
            if len(fc) and np.allclose(fc[:, :3], _SHAP_GREY, atol=0.01):
                coll.set_facecolor(MISSING_COLOR)
                coll.set_edgecolor("none")
    for cax in fig.axes:
        if cax not in flat:
            cax.tick_params(labelsize=PT_TICK, length=2)
            cax.set_ylabel(cax.get_ylabel(), fontsize=PT_TICK + 0.5, labelpad=2)
