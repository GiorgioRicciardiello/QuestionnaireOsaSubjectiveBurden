"""Tests for the standalone main-figure panels (AUROC bars + SHAP bars)."""

import pandas as pd
import pytest

from config import PATHS
from questionnaire_osa.plots.main_panels import (
    AHI_ENDPOINTS,
    FEATURE_SETS,
    load_auroc_panel,
    load_shap_panels,
    parse_estimate,
    plot_auroc_grouped_bar,
    plot_importance_barh,
)

DELONG_CSV = PATHS["tbl_model_performance"] / "xgb_auroc_delong.csv"
SHAP_CSV = PATHS["tbl_shap_binary"]

needs_tables = pytest.mark.skipif(
    not (DELONG_CSV.exists() and SHAP_CSV.exists()),
    reason="result tables not present in this environment",
)


def test_parse_estimate_reads_point_and_en_dash_interval():
    est = parse_estimate("0.764 (0.752–0.776)")
    assert (est.point, est.lo, est.hi) == (0.764, 0.752, 0.776)


def test_parse_estimate_rejects_malformed_text():
    with pytest.raises(ValueError):
        parse_estimate("0.764")


@needs_tables
def test_auroc_values_match_delong_table():
    panel = load_auroc_panel(DELONG_CSV)
    assert panel.endpoints == AHI_ENDPOINTS
    assert [e.point for e in panel.auroc["All Features"]] == [0.764, 0.743, 0.763]
    assert [e.point for e in panel.auroc["Questionnaire Only"]] == [0.690, 0.674, 0.699]
    assert [e.point for e in panel.auroc["Demographics Only"]] == [0.749, 0.718, 0.726]


@needs_tables
def test_delta_uses_paired_delong_difference_not_rounded_subtraction():
    # 0.764 - 0.749 = 0.015, but the manuscript reports the paired DeLong
    # difference (0.016-0.038), computed before rounding.
    panel = load_auroc_panel(DELONG_CSV)
    assert [d.point for d in panel.delta] == [0.016, 0.025, 0.038]


@needs_tables
def test_shap_panels_split_demographics_from_questionnaire_items():
    demo, quest = load_shap_panels(SHAP_CSV, top_n=5)
    assert [f for f, _ in demo] == ["BMI", "Age", "Gender", "Race"]
    assert [f for f, _ in quest][:3] == ["Snoring", "Witnessed Apneas", "High Blood Pressure"]
    assert len(quest) == 5
    assert not {f for f, _ in quest} & {"Age", "BMI", "Gender", "Race", "Obese"}


@needs_tables
def test_shap_value_is_mean_over_the_three_ahi_thresholds():
    df = pd.read_csv(SHAP_CSV)
    rows = df[(df.Feature_Set == "All Features") & (df.Feature == "Snoring")
              & df.Target.isin(["AHI @ 5", "AHI @ 15", "AHI @ 30"])]
    _, quest = load_shap_panels(SHAP_CSV, top_n=5)
    assert dict(quest)["Snoring"] == pytest.approx(rows["Mean_|SHAP|"].mean())


@needs_tables
def test_plots_write_png_svg_pdf(tmp_path):
    panel = load_auroc_panel(DELONG_CSV)
    demo, _ = load_shap_panels(SHAP_CSV, top_n=5)
    written = plot_auroc_grouped_bar(panel, tmp_path / "auroc")
    written += plot_importance_barh(demo, "Top demographic predictors", "#79A9D8",
                                    tmp_path / "demo")
    assert sorted(p.suffix for p in written) == [".pdf", ".pdf", ".png", ".png", ".svg", ".svg"]
    assert all(p.stat().st_size > 0 for p in written)


def test_feature_set_order_matches_legend_order():
    assert FEATURE_SETS == ("All Features", "Questionnaire Only", "Demographics Only")
