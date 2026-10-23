"""Entry point: redraw manuscript Figure 1 (ROC grid) from a stored run.

Run with::

    python -m questionnaire_osa.manuscript.make_figure1 [--pkl PATH] [--out STEM]

Defaults read ``results/training/xgboost_cv_results.pkl`` and write
``results/figures/model_performance/figure1_roc.{pdf,svg,png,tiff}``.  No
model is retrained.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from config import PATHS

from ..plots.roc_figure import plot_roc_figure


DEFAULT_PKL = PATHS["xgb_results_pkl"]
DEFAULT_STEM = PATHS["fig_model_performance"] / "figure1_roc"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--out", type=Path, default=DEFAULT_STEM)
    args = parser.parse_args()

    if not args.pkl.exists():
        raise FileNotFoundError(f"Training artifacts not found at {args.pkl}")
    with args.pkl.open("rb") as fh:
        artifacts = pickle.load(fh)

    written = plot_roc_figure(
        artifacts["feature_sets"], artifacts["binary_targets"],
        save_stem=args.out,
    )
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
