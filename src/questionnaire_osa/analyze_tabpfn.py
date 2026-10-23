"""Entry point: regenerate ROC curves and confusion matrices for TabPFN.

Run with::

    python -m questionnaire_osa.analyze_tabpfn

TabPFN is evaluated with k-fold CV (no held-out test), so we:

1. Concatenate all out-of-fold predictions into a single y_true/y_proba
   stream per target.
2. Plot the resulting ROC curve (binary) and aggregated confusion matrix.

These are exploratory diagnostics, not referenced by the manuscript or
supplementary docx (the manuscript's TabPFN confusion figure, Figure 2, is
regenerated separately by ``manuscript.make_figure2``), so they are routed
to the gitignored draft archive.

Output
------
* ``_local/figures_archive/draft_regen/tabpfn/roc_curves.png``
* ``_local/figures_archive/draft_regen/tabpfn/confusion_matrices_binary.png``
* ``_local/figures_archive/draft_regen/tabpfn/confusion_matrices_multiclass.png``
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from config import PATHS, SETTINGS
from config.config import ensure_output_dirs

from .plots import add_panel_labels, apply_publication_style
from .utils import set_global_seeds


def _load_artifacts(path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"TabPFN artifacts not found at {path}\n"
            "Run `python -m questionnaire_osa.train_tabpfn` first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


def _concat_oof(
    fold_predictions: Sequence[tuple],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate per-fold ``(y_true, y_pred, y_proba)`` into single arrays."""
    y_true  = np.concatenate([fp[0] for fp in fold_predictions])
    y_pred  = np.concatenate([fp[1] for fp in fold_predictions])
    y_proba = np.concatenate([fp[2] for fp in fold_predictions])
    return y_true, y_pred, y_proba


# ── Plots ─────────────────────────────────────────────────────────────────────
def _plot_roc(
    artifacts: Dict[str, Any],
    binary_targets: Sequence[str],
    save_path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    targets_config = artifacts["targets_config"]

    for idx, target in enumerate(binary_targets):
        ax = axes.flatten()[idx]
        y_true, _, y_proba = _concat_oof(artifacts["all_predictions"][target])
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)

        ax.plot(fpr, tpr, color="#0072B2", lw=2, label=f"AUROC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
        ax.set_title(f"{desc}\n(n = {len(y_true):,})", fontsize=10, fontweight="bold")
        ax.set_xlabel("1 − Specificity")
        ax.set_ylabel("Sensitivity")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    plt.suptitle("TabPFN v2.5 — ROC Curves (out-of-fold predictions)",
                 fontsize=12, fontweight="bold")
    add_panel_labels(axes, x=-0.08, y=1.08)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(
    artifacts: Dict[str, Any],
    targets: Sequence[str],
    labels: Sequence[str],
    suptitle: str,
    save_path,
    grid: tuple[int, int],
    cell_size: tuple[float, float],
) -> None:
    rows, cols = grid
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * cell_size[0], rows * cell_size[1]),
    )
    axes_flat = np.atleast_1d(axes).flatten()
    targets_config = artifacts["targets_config"]

    for idx, target in enumerate(targets):
        y_true, y_pred, _ = _concat_oof(artifacts["all_predictions"][target])
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        annot = np.array([
            [f"{c}\n({p:.1f}%)" for c, p in zip(row_c, row_p)]
            for row_c, row_p in zip(cm, cm_pct)
        ])
        ax = axes_flat[idx]
        sns.heatmap(
            cm_pct, annot=annot, fmt="", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
            cbar=False, vmin=0, vmax=100, linewidths=2, linecolor="white",
            annot_kws={"fontsize": 9},
        )
        desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
        ax.set_title(f"{desc}\n(n = {len(y_true):,})", fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle(suptitle, fontsize=12, fontweight="bold")
    add_panel_labels(np.array(axes), x=-0.08, y=1.08)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    set_global_seeds(SETTINGS["random_state"])
    ensure_output_dirs()
    apply_publication_style()

    artifacts = _load_artifacts(PATHS["tabpfn_results_pkl"])
    binary_targets     = artifacts["binary_targets"]
    multiclass_targets = artifacts["multiclass_targets"]

    print(f"Loaded TabPFN artifacts: {artifacts['n_samples']:,} samples × "
          f"{artifacts['n_features']} features, {artifacts['n_folds']}-fold CV")

    _plot_roc(
        artifacts, binary_targets,
        save_path=PATHS["fig_tabpfn"] / "roc_curves.png",
    )
    _plot_confusion(
        artifacts, binary_targets, ["No OSA", "OSA"],
        suptitle="TabPFN v2.5 — Confusion Matrices (Binary, OOF)",
        save_path=PATHS["fig_tabpfn"] / "confusion_matrices_binary.png",
        grid=(2, 3), cell_size=(4.0, 3.5),
    )
    _plot_confusion(
        artifacts, multiclass_targets, ["None", "Mild", "Moderate", "Severe"],
        suptitle="TabPFN v2.5 — Confusion Matrices (4-class, OOF)",
        save_path=PATHS["fig_tabpfn"] / "confusion_matrices_multiclass.png",
        grid=(1, 2), cell_size=(4.5, 4.0),
    )

    print(f"TabPFN figures → {PATHS['fig_tabpfn']}")


if __name__ == "__main__":
    main()
