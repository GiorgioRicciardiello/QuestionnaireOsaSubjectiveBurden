"""Entry point: draw the standalone panels for the hand-assembled main figure.

Run with::

    python -m questionnaire_osa.manuscript.make_main_figure_panels [--out DIR] [--top-n N]

Reads ``results/tables/model_performance/xgb_auroc_delong.csv`` and
``results/tables/feature_importance/shap_feature_importance_binary.csv`` and
writes ``auroc_grouped_bar``, ``demographic_predictors_barh`` and
``questionnaire_predictors_barh`` (.png/.svg/.pdf) to
``results/figures/main_figure_panels/``.  No model is retrained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import PATHS

from ..plots.main_panels import (
    COLORS,
    load_auroc_panel,
    load_shap_panels,
    plot_auroc_grouped_bar,
    plot_importance_barh,
)


DELONG_CSV = PATHS["tbl_model_performance"] / "xgb_auroc_delong.csv"
SHAP_CSV = PATHS["tbl_shap_binary"]
DEFAULT_DIR = PATHS["fig_main_panels"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()

    for path in (DELONG_CSV, SHAP_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Result table not found at {path}")

    panel = load_auroc_panel(DELONG_CSV)
    demographics, questionnaire = load_shap_panels(SHAP_CSV, top_n=args.top_n)

    written = plot_auroc_grouped_bar(panel, args.out / "auroc_grouped_bar")
    written += plot_importance_barh(
        demographics, "Top demographic predictors",
        COLORS["Demographics Only"], args.out / "demographic_predictors_barh",
    )
    written += plot_importance_barh(
        questionnaire, "Most informative questionnaire items",
        COLORS["Questionnaire Only"], args.out / "questionnaire_predictors_barh",
    )
    for path in written:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
