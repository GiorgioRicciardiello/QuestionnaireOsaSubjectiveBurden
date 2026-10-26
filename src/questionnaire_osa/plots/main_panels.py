"""Standalone panels for the hand-assembled main figure.

Three separate figures (not subplots), each read from the result tables so
no number is typed in by hand:

* grouped AUROC bars for AHI >=5/15/30 by feature set, with 95% CIs and the
  paired-DeLong All Features - Demographics Only difference;
* mean |SHAP| of the demographic predictors;
* mean |SHAP| of the top questionnaire items.

SHAP values are from the All Features model, averaged over the three AHI
thresholds, so values in the two importance panels are directly comparable
(each panel still sets its own x-axis range).  Text is kept as
text in the SVG/PDF output for editing in Illustrator or draw.io.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


AHI_ENDPOINTS: Tuple[str, ...] = ("AHI ≥5", "AHI ≥15", "AHI ≥30")
SHAP_TARGETS: Tuple[str, ...] = ("AHI @ 5", "AHI @ 15", "AHI @ 30")
FEATURE_SETS: Tuple[str, ...] = ("All Features", "Questionnaire Only", "Demographics Only")

COLORS: Dict[str, str] = {
    "All Features":       "#8E79B7",   # muted purple
    "Questionnaire Only": "#F2B37C",   # soft orange
    "Demographics Only":  "#79A9D8",   # muted blue
}
INK = "#1F1F1F"
GRID = "#D9D9D9"

_DELTA_COL = "All Features − Demographics Only (paired DeLong 95% CI)"
_ESTIMATE_RE = re.compile(r"^\s*([\d.]+)\s*\(\s*([\d.]+)\s*[–-]\s*([\d.]+)\s*\)\s*$")

_PANEL_RC = {
    "font.family":       ["Arial", "DejaVu Sans"],
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.titlelocation": "left",
    "axes.labelsize":    9,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "axes.edgecolor":    INK,
    "text.color":        INK,
    "axes.labelcolor":   INK,
    "xtick.color":       INK,
    "ytick.color":       INK,
    "svg.fonttype":      "none",   # keep text editable
    "pdf.fonttype":      42,
    "savefig.dpi":       300,
}


@dataclass(frozen=True)
class Estimate:
    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class AurocPanel:
    endpoints: Tuple[str, ...]
    auroc: Dict[str, Tuple[Estimate, ...]]
    delta: Tuple[Estimate, ...]


def parse_estimate(text: str) -> Estimate:
    """Parse ``"0.764 (0.752–0.776)"`` into an :class:`Estimate`."""
    match = _ESTIMATE_RE.match(str(text))
    if match is None:
        raise ValueError(f"Not an 'estimate (lo–hi)' string: {text!r}")
    point, lo, hi = (float(g) for g in match.groups())
    return Estimate(point, lo, hi)


def load_auroc_panel(delong_csv: Path) -> AurocPanel:
    """AUROCs and paired DeLong differences for the AHI endpoints."""
    df = pd.read_csv(delong_csv).set_index("Endpoint")
    missing = [e for e in AHI_ENDPOINTS if e not in df.index]
    if missing:
        raise KeyError(f"Endpoints missing from {delong_csv}: {missing}")
    rows = df.loc[list(AHI_ENDPOINTS)]
    auroc = {
        fs: tuple(parse_estimate(v) for v in rows[f"{fs} AUROC (95% CI)"])
        for fs in FEATURE_SETS
    }
    delta = tuple(parse_estimate(v) for v in rows[_DELTA_COL])
    return AurocPanel(AHI_ENDPOINTS, auroc, delta)


def load_shap_panels(
    shap_csv: Path, top_n: int = 5,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Mean |SHAP| (All Features model, averaged over AHI thresholds).

    Returns ``(demographics, questionnaire_top_n)``, each sorted largest
    first.  Group membership comes from the Demographics Only and
    Questionnaire Only feature sets, so derived BMI-category flags (e.g.
    "Obese") fall in neither panel.
    """
    df = pd.read_csv(shap_csv)
    demo_feats = set(df.loc[df.Feature_Set == "Demographics Only", "Feature"])
    quest_feats = set(df.loc[df.Feature_Set == "Questionnaire Only", "Feature"])

    rows = df[(df.Feature_Set == "All Features") & df.Target.isin(SHAP_TARGETS)]
    counts = rows.groupby("Feature")["Target"].nunique()
    incomplete = counts[counts != len(SHAP_TARGETS)]
    if not incomplete.empty:
        raise ValueError(f"Features missing a target in {shap_csv}: {list(incomplete.index)}")
    means = rows.groupby("Feature")["Mean_|SHAP|"].mean().sort_values(ascending=False)

    demographics = [(f, float(v)) for f, v in means.items() if f in demo_feats]
    questionnaire = [(f, float(v)) for f, v in means.items() if f in quest_feats][:top_n]
    return demographics, questionnaire


def _style_axes(ax: plt.Axes, grid_axis: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    ax.patch.set_alpha(0.0)


def _save(fig: plt.Figure, stem: Path) -> List[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "svg", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight", transparent=True)
        written.append(path)
    plt.close(fig)
    return written


def plot_auroc_grouped_bar(
    panel: AurocPanel, stem: Path, figsize: Tuple[float, float] = (4.6, 3.1),
) -> List[Path]:
    """Grouped AUROC bars with 95% CI whiskers and Δ brackets."""
    with plt.rc_context(_PANEL_RC):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(panel.endpoints))
        width = 0.25
        offsets = dict(zip(FEATURE_SETS, (-width, 0.0, width)))

        for fs in FEATURE_SETS:
            ests = panel.auroc[fs]
            pts = np.array([e.point for e in ests])
            err = np.array([[e.point - e.lo for e in ests], [e.hi - e.point for e in ests]])
            xs = x + offsets[fs]
            ax.bar(xs, pts, width=width * 0.9, color=COLORS[fs], label=fs,
                   yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.8,
                                       "capthick": 0.6, "ecolor": INK})
            for xi, e in zip(xs, ests):
                ax.text(xi, e.hi + 0.006, f"{e.point:.3f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK)

        for i, d in enumerate(panel.delta):
            top = max(e.hi for fs in FEATURE_SETS for e in [panel.auroc[fs][i]])
            y = top + 0.05
            x1, x2 = x[i] + offsets["All Features"], x[i] + offsets["Demographics Only"]
            ax.plot([x1, x1, x2, x2], [y - 0.01, y, y, y - 0.01], color=INK, lw=0.6)
            ax.text((x1 + x2) / 2, y + 0.005, f"Δ = {d.point:.3f}", ha="center",
                    va="bottom", fontsize=7.5, color=INK)

        ax.set_title("Test set performance (AUROC)", pad=8)
        ax.set_ylabel("AUROC")
        ax.set_xticks(x)
        ax.set_xticklabels(panel.endpoints, fontweight="bold")
        ax.set_ylim(0.4, 1.0)
        _style_axes(ax, "y")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                  ncol=len(FEATURE_SETS), handlelength=1.0, handleheight=0.8,
                  columnspacing=1.2, handletextpad=0.4)
        fig.tight_layout()
        return _save(fig, stem)


def plot_importance_barh(
    items: Sequence[Tuple[str, float]],
    title: str,
    color: str,
    stem: Path,
    axes_width: float = 2.0,
    bar_pitch: float = 0.24,
) -> List[Path]:
    """Horizontal mean |SHAP| bars, largest at the top.

    The plot area is fixed in inches (``axes_width`` wide, ``bar_pitch``
    per bar) so every panel has the same bar thickness and plot width
    regardless of label length; labels and title extend the saved canvas.
    """
    if not items:
        raise ValueError("No features to plot")
    labels = [f for f, _ in items][::-1]
    values = np.array([v for _, v in items][::-1])
    xmax = float(np.ceil(values.max() * 1.2 * 20) / 20)   # round up to 0.05

    with plt.rc_context(_PANEL_RC):
        fig = plt.figure(figsize=(axes_width, bar_pitch * len(values)))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.barh(labels, values, color=color, height=0.68)
        for yi, v in enumerate(values):
            ax.text(v + xmax * 0.015, yi, f"{v:.2f}", va="center", ha="left",
                    fontsize=7.5, color=INK)
        ax.set_ylim(-0.55, len(values) - 0.45)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.set_title(title, pad=6)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        _style_axes(ax, "x")
        return _save(fig, stem)
