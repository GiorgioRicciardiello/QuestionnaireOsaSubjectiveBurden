"""Tests for the public-release shared-dataset builder."""

import pandas as pd
import pytest

from config.config import FEATURE_PREFIXES, TARGETS
from questionnaire_osa.data import build_feature_sets
from questionnaire_osa.manuscript.make_shared_dataset import (
    AGE_TOP_CODE,
    INDEX_COLUMNS,
    build_data_dictionary,
    build_shared_dataframe,
)

pytestmark = pytest.mark.skipif(
    not pytest.importorskip("config.config").PATHS["database"].exists(),
    reason="sleep_research_cleaned.db not present in this environment",
)


@pytest.fixture(scope="module")
def shared() -> pd.DataFrame:
    return build_shared_dataframe()


def test_row_count_matches_analytical_cohort(shared: pd.DataFrame) -> None:
    assert len(shared) == 67_397


def test_subject_id_is_fresh_sequential_and_unique(shared: pd.DataFrame) -> None:
    assert shared["subject_id"].is_unique
    assert shared["subject_id"].min() == 1
    assert shared["subject_id"].max() == len(shared)


def test_contains_all_169_feature_columns(shared: pd.DataFrame) -> None:
    feature_sets = build_feature_sets(shared, FEATURE_PREFIXES)
    # Re-deriving by prefix on the exported frame should recover the same
    # 169 all-features columns used to train the published models.
    assert len(feature_sets["all_features"]["cols"]) == 169


def test_contains_all_8_target_columns(shared: pd.DataFrame) -> None:
    for spec in TARGETS.values():
        assert spec["col"] in shared.columns


def test_contains_renamed_severity_indices(shared: pd.DataFrame) -> None:
    for new_name in INDEX_COLUMNS.values():
        assert new_name in shared.columns
        assert shared[new_name].dtype.kind == "f"


def test_drops_internal_identifiers_and_unused_columns(shared: pd.DataFrame) -> None:
    dropped = {
        "id", "study_year", "LAB_CAT", "MD_Identifyer",
        "CPAP", "BIPAP", "Oxygen",
        "eng_is_ep_reading_missing",
    }
    assert dropped.isdisjoint(shared.columns)


def test_demographics_match_published_manuscript_figures(shared: pd.DataFrame) -> None:
    # Uncapped mean is 54.252; top-coding 137 ages to 90 lowers it to 54.249,
    # just outside +/-0.05 of the rounded published 54.3.
    assert shared["dem_age"].mean() == pytest.approx(54.3, abs=0.06)
    assert shared["dem_age"].std() == pytest.approx(15.6, abs=0.05)
    assert 100 * shared["dem_gender"].mean() == pytest.approx(46.7, abs=0.05)


def test_data_dictionary_has_one_row_per_column_with_no_blank_definitions(
    shared: pd.DataFrame,
) -> None:
    data_dict = build_data_dictionary(shared)
    assert len(data_dict) == shared.shape[1]
    assert set(data_dict["column"]) == set(shared.columns)
    assert (data_dict["definition"].fillna("").str.len() > 0).all()


def test_ages_above_89_are_top_coded_to_90(shared: pd.DataFrame) -> None:
    # HIPAA Safe Harbor: ages over 89 must be aggregated into one 90+ category.
    assert shared["dem_age"].max() == AGE_TOP_CODE == 90
    assert (shared["dem_age"] == AGE_TOP_CODE).sum() >= 137


def test_data_dictionary_labels_age_top_code(shared: pd.DataFrame) -> None:
    data_dict = build_data_dictionary(shared).set_index("column")
    assert "90=90 years or older" in data_dict.loc["dem_age", "encoding"]
