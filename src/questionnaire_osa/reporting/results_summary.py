"""Render result summaries as console text and CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd


# ── Display order ─────────────────────────────────────────────────────────────
_FS_DISPLAY: Dict[str, str] = {
    "all_features": "All Features",
    "dem_only":     "Demographics Only",
    "no_dem":       "Questionnaire Only",
}

_BINARY_KEYS: list[tuple[str, str]] = [
    ("AHI ≥ 5",  "ahi_5"),
    ("AHI ≥ 15", "ahi_15"),
    ("AHI ≥ 30", "ahi_30"),
    ("RDI ≥ 5",  "rdi_5"),
    ("RDI ≥ 15", "rdi_15"),
    ("RDI ≥ 30", "rdi_30"),
]

_METRICS: list[tuple[str, str]] = [
    ("AUROC",       "auroc"),
    ("Precision",   "precision"),
    ("Sensitivity", "recall"),
    ("Specificity", "specificity"),
    ("F1",          "f1"),
]


def print_results_summary(
    feature_sets: Dict[str, Any],
    fs_order: Sequence[str] = ("all_features", "no_dem", "dem_only"),
) -> None:
    """Pretty-print binary and 4-class results for each feature set."""
    width = 82

    for fs_name in fs_order:
        r = feature_sets[fs_name]["all_results"]
        n_features = feature_sets[fs_name]["n_features"]

        print(f"\n{_FS_DISPLAY[fs_name].upper()} ({n_features} features)")
        print("=" * width)

        header = f"{'':>13}" + "".join(f"{name:>11}" for name, _ in _BINARY_KEYS)
        print(header)
        print("-" * width)
        for metric_label, metric_key in _METRICS:
            vals = "".join(f"{r[k][metric_key]:>11.4f}" for _, k in _BINARY_KEYS)
            print(f"{metric_label:>13}{vals}")

        print()
        print(f"{'4-CLASS':>13} {'AHI':>11} {'RDI':>11}")
        print("-" * 37)
        print(f"{'Macro F1':>13} {r['ahi_4class']['macro_f1']:>11.4f} {r['rdi_4class']['macro_f1']:>11.4f}")
        print(f"{'Accuracy':>13} {r['ahi_4class']['accuracy']:>11.4f} {r['rdi_4class']['accuracy']:>11.4f}")


def build_results_summary_csv(
    feature_sets: Dict[str, Any],
    output_path: Path,
) -> pd.DataFrame:
    """Build a long-form CSV: ``(feature_set, target, metric, value)``.

    Suitable for downstream filtering / pivoting in the manuscript.
    """
    rows: list[dict[str, Any]] = []

    for fs_name, fs_block in feature_sets.items():
        results = fs_block["all_results"]
        n_features = fs_block["n_features"]
        for target_name, target_metrics in results.items():
            for metric, value in target_metrics.items():
                if metric == "confusion_matrix":
                    continue
                rows.append({
                    "feature_set": fs_name,
                    "n_features":  n_features,
                    "target":      target_name,
                    "metric":      metric,
                    "value":       float(value),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
