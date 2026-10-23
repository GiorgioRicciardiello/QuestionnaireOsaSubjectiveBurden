"""Decision curve figure: net benefit per endpoint, with paired-difference panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ENDPOINT_LABELS = {
    "ahi_5": "AHI ≥5", "ahi_15": "AHI ≥15", "ahi_30": "AHI ≥30",
    "rdi_5": "RDI ≥5", "rdi_15": "RDI ≥15", "rdi_30": "RDI ≥30",
}
MODEL_STRATEGIES = ("All-Features", "Demographics-Only", "Questionnaire-Only")
STRATEGY_STYLE = {
    "All-Features": dict(color="#1f4e79", lw=2),
    "Demographics-Only": dict(color="#c55a11", lw=2),
    "Questionnaire-Only": dict(color="#7f7f7f", lw=1.5, ls="--"),
    "treat_all": dict(color="black", lw=1, ls=":"),
    "treat_none": dict(color="black", lw=0.8),
}


def plot_decision_curves(curves: pd.DataFrame, boot: pd.DataFrame, stem: Path) -> list[Path]:
    """Save a 2x6 grid: top row net benefit, bottom row All minus Demographics-Only."""
    endpoints = list(ENDPOINT_LABELS)
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.2), sharex=True)
    for col, ep in enumerate(endpoints):
        ax = axes[0, col]
        sub = curves[curves.endpoint == ep]
        for strat, style in STRATEGY_STYLE.items():
            s = sub[sub.strategy == strat]
            ax.plot(s.threshold, s.net_benefit, label=strat, **style)
        ax.set_title(ENDPOINT_LABELS[ep])
        models = sub[sub.strategy.isin(MODEL_STRATEGIES)]
        top = max(0.05, float(sub.net_benefit.max()) + 0.02)
        bottom = min(-0.05, float(models.net_benefit.min()) - 0.01)
        ax.set_xlim(0, 0.80)
        ax.set_ylim(bottom, top)
        ax.grid(alpha=0.3)
        if col == 0:
            ax.set_ylabel("Net benefit")

        bx = axes[1, col]
        d = boot[(boot.endpoint == ep) & (boot.contrast == "All-Features minus Demographics-Only")]
        bx.errorbar(d.threshold, d.delta_nb,
                    yerr=[d.delta_nb - d.ci_lo, d.ci_hi - d.delta_nb],
                    fmt="o", color="#1f4e79", capsize=4)
        bx.axhline(0, color="black", lw=0.8)
        bx.set_xticks([0.10, 0.20, 0.30])
        bx.grid(alpha=0.3)
        bx.set_xlabel("Threshold probability")
        if col == 0:
            bx.set_ylabel("All − Demographics-Only\n(paired, 95% CI)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Decision curves, provisional (uncalibrated probabilities)", y=0.99)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)

    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        path = stem.with_suffix(f".{ext}")
        fig.savefig(path, dpi=200 if ext == "png" else None, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written
