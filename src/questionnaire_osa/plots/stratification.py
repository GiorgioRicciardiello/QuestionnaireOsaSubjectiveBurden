"""Stratification verification plots.

Compare the distribution of every stratification variable (and the
target itself) across the *full / train_val / test* splits.  A small
*max-deviation* annotation on each panel makes it easy to flag any
mis-balanced variable at a glance.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import add_panel_labels


# ── Helpers ───────────────────────────────────────────────────────────────────
def _bin_continuous(series: pd.Series, bins: List[float]) -> pd.Series:
    """Bin a continuous series with the same labels used for stratification."""
    labels = (
        [f"<{bins[1]}"]
        + [f"{bins[i]}-{bins[i + 1]}" for i in range(1, len(bins) - 2)]
        + [f">={bins[-2]}"]
    )
    return pd.cut(series, bins=bins, labels=labels)


def _distribution_pct(
    series: pd.Series, top_n: int | None = None,
) -> pd.Series:
    """Percentage distribution of values, sorted by index."""
    counts = series.value_counts(normalize=True) * 100
    if top_n is not None:
        counts = counts.head(top_n)
    return counts.sort_index()


# ── Single-panel plot ─────────────────────────────────────────────────────────
def plot_distribution_comparison(
    df_full: pd.DataFrame,
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    col: str,
    ax: plt.Axes,
    title: str,
    bins: List[float] | None = None,
    top_n: int | None = None,
) -> Tuple[float, float]:
    """Grouped-bar comparison of one variable's distribution across splits.

    Returns
    -------
    (train_dev, test_dev)
        Maximum percentage-point deviation of train_val vs test compared
        to the full dataset.
    """
    if bins is not None:
        full  = _bin_continuous(df_full[col],      bins)
        train = _bin_continuous(df_train_val[col], bins)
        test  = _bin_continuous(df_test[col],      bins)
    else:
        full  = df_full[col].fillna(-1).astype(int)
        train = df_train_val[col].fillna(-1).astype(int)
        test  = df_test[col].fillna(-1).astype(int)

    full_pct  = _distribution_pct(full,  top_n)
    train_pct = _distribution_pct(train, top_n)
    test_pct  = _distribution_pct(test,  top_n)

    cats = sorted(set(full_pct.index) | set(train_pct.index) | set(test_pct.index))
    full_pct  = full_pct.reindex(cats,  fill_value=0)
    train_pct = train_pct.reindex(cats, fill_value=0)
    test_pct  = test_pct.reindex(cats,  fill_value=0)

    train_dev = float((train_pct - full_pct).abs().max())
    test_dev  = float((test_pct  - full_pct).abs().max())
    max_dev   = max(train_dev, test_dev)

    x = np.arange(len(cats))
    w = 0.25
    ax.bar(x - w, full_pct.values,  w,
           label=f"Full (N={len(df_full):,})",      color="#0072B2", alpha=0.8)
    ax.bar(x,     train_pct.values, w,
           label=f"Train/Val (N={len(df_train_val):,})", color="#E69F00", alpha=0.8)
    ax.bar(x + w, test_pct.values,  w,
           label=f"Test (N={len(df_test):,})",      color="#009E73", alpha=0.8)

    ax.set_title(f"{title} (max Δ: {max_dev:.2f}pp)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:8] for c in cats], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Distribution (%)")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    return train_dev, test_dev


# ── Multi-panel wrappers ──────────────────────────────────────────────────────
StratVarSpec = Tuple[str, str, List[float] | None, int | None]
"""``(column, title, bins_or_None, top_n_or_None)`` per variable."""


def plot_strata_panel(
    df_full: pd.DataFrame,
    df_train_val: pd.DataFrame,
    df_test: pd.DataFrame,
    strat_vars: Sequence[StratVarSpec],
    suptitle: str,
    save_path: Path,
) -> None:
    """2×4 grid of distribution-comparison panels (8 stratification vars)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for idx, (col, title, bins, top_n) in enumerate(strat_vars):
        plot_distribution_comparison(
            df_full, df_train_val, df_test,
            col, axes.flatten()[idx], title, bins, top_n,
        )
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    add_panel_labels(axes)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_target_distribution(
    y_full: pd.Series,
    y_train_val: pd.Series,
    y_test: pd.Series,
    binary_labels: Sequence[str],
    multi_y_full: pd.Series,
    multi_y_train_val: pd.Series,
    multi_y_test: pd.Series,
    multi_labels: Sequence[str],
    save_path: Path,
) -> None:
    """1×2 figure: binary + multiclass target distribution across splits."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Binary
    ax = axes[0]
    full_pct  = (y_full.value_counts(normalize=True).sort_index()      * 100)
    train_pct = (y_train_val.value_counts(normalize=True).sort_index() * 100)
    test_pct  = (y_test.value_counts(normalize=True).sort_index()      * 100)
    max_dev   = max(
        (train_pct - full_pct).abs().max(),
        (test_pct  - full_pct).abs().max(),
    )
    x, w = np.arange(len(binary_labels)), 0.25
    ax.bar(x - w, full_pct.values,  w,
           label=f"Full (N={len(y_full):,})",      color="#0072B2", alpha=0.8)
    ax.bar(x,     train_pct.values, w,
           label=f"Train/Val (N={len(y_train_val):,})", color="#E69F00", alpha=0.8)
    ax.bar(x + w, test_pct.values,  w,
           label=f"Test (N={len(y_test):,})",      color="#009E73", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(binary_labels)
    ax.set_ylabel("Distribution (%)")
    ax.set_title(f"Binary Target  (max Δ: {max_dev:.2f}pp)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Multiclass
    ax = axes[1]
    full_pct  = (multi_y_full.value_counts(normalize=True).sort_index()      * 100)
    train_pct = (multi_y_train_val.value_counts(normalize=True).sort_index() * 100)
    test_pct  = (multi_y_test.value_counts(normalize=True).sort_index()      * 100)
    max_dev   = max(
        (train_pct - full_pct).abs().max(),
        (test_pct  - full_pct).abs().max(),
    )
    x = np.arange(len(multi_labels))
    ax.bar(x - w, full_pct.values,  w,
           label=f"Full (N={len(multi_y_full):,})",      color="#0072B2", alpha=0.8)
    ax.bar(x,     train_pct.values, w,
           label=f"Train/Val (N={len(multi_y_train_val):,})", color="#E69F00", alpha=0.8)
    ax.bar(x + w, test_pct.values,  w,
           label=f"Test (N={len(multi_y_test):,})",      color="#009E73", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(multi_labels)
    ax.set_ylabel("Distribution (%)")
    ax.set_title(f"Multiclass Target (max Δ: {max_dev:.2f}pp)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Target Class Distribution Across Splits",
                 fontsize=14, fontweight="bold")
    add_panel_labels(axes)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
