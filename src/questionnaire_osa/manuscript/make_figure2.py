"""Entry point: redraw manuscript Figure 2 (TabPFN confusion matrices).

Run with::

    python -m questionnaire_osa.manuscript.make_figure2 [--pkl PATH] [--out STEM] [--counts mean|sum]

Defaults read ``results/training/tabpfn_cv_results.pkl`` and write
``results/figures/model_performance/figure2_confusion.{pdf,svg,png,tiff}``.
Nothing is retrained.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from config import PATHS

from ..plots.confusion_figure import TARGETS, plot_confusion_figure


DEFAULT_PKL = PATHS["tabpfn_results_pkl"]
DEFAULT_STEM = PATHS["fig_model_performance"] / "figure2_confusion"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    parser.add_argument("--out", type=Path, default=DEFAULT_STEM)
    parser.add_argument("--counts", choices=("mean", "sum"), default="mean",
                        help="mean of the 5 folds (n = 13,479) or their sum")
    args = parser.parse_args()

    if not args.pkl.exists():
        raise FileNotFoundError(f"Training artifacts not found at {args.pkl}")
    with args.pkl.open("rb") as fh:
        artifacts = pickle.load(fh)

    results = artifacts["all_results"]
    folds = {t: np.asarray(results[t]["confusion_matrix"])
             for row in TARGETS for t in row}
    for path in plot_confusion_figure(folds, args.out, counts=args.counts):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
