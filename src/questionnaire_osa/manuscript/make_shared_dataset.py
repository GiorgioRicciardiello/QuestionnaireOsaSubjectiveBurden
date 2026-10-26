"""Build the de-identified, analysis-only dataset for public data sharing.

Keeps exactly the rows and columns that entered the published models:
the adults-only cohort (``dem_age >= 18``, n = 67,397) and the 169
``all_features`` predictor columns plus the 8 target columns and the 2
continuous severity indices used to derive them. Internal identifiers
(record id, lab/physician codes, study year, treatment flags) and
columns never fed to a model (missingness-indicator engineering columns)
are dropped, and ages over 89 are top-coded to 90. Row order is shuffled with the project's standard random
seed before assigning a fresh sequential ``subject_id`` so the release
order carries no information about collection order or site.

Run with ``python -m questionnaire_osa.manuscript.make_shared_dataset``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from config.config import FEATURE_PREFIXES, PATHS, SETTINGS, TARGETS
from ..data import build_feature_sets, load_sleep_data

# Continuous severity indices that the 8 binary/four-class targets were
# derived from (Methods 2.2, Outcome Definitions). Renamed on export to
# avoid the hyphenated raw column names.
INDEX_COLUMNS: Dict[str, str] = {
    "resp-ahi_no_reras_index-total": "resp_ahi_total",
    "resp-rdi_with_reras_index-total": "resp_rdi_total",
}

CLASS_NAMES: List[str] = ["No OSA", "Mild OSA", "Moderate OSA", "Severe OSA"]

# HIPAA Safe Harbor: ages over 89 are aggregated into a single 90+ category
# on export. The published models were trained on the uncapped ages, so this
# affects only the released file (137 subjects aged 90-99).
AGE_TOP_CODE: int = 90


def _load_encoding() -> Dict[str, Any]:
    with open(PATHS["encoding_json"], encoding="utf-8") as fh:
        return json.load(fh)


def build_shared_dataframe() -> pd.DataFrame:
    """Return the de-identified analytical dataset, with a fresh subject_id."""
    df = load_sleep_data(PATHS["database"], min_age=SETTINGS["min_age"])
    feature_sets = build_feature_sets(df, FEATURE_PREFIXES)
    feature_cols = feature_sets["all_features"]["cols"]
    target_cols = [spec["col"] for spec in TARGETS.values()]

    keep_cols = feature_cols + target_cols + list(INDEX_COLUMNS)
    shared = df[keep_cols].rename(columns=INDEX_COLUMNS).copy()
    shared["dem_age"] = shared["dem_age"].clip(upper=AGE_TOP_CODE)

    shared = shared.sample(frac=1, random_state=SETTINGS["random_state"]).reset_index(drop=True)
    shared.insert(0, "subject_id", range(1, len(shared) + 1))
    return shared


def build_data_dictionary(shared: pd.DataFrame) -> pd.DataFrame:
    """Return one row per shared column: definition, encoding, dtype, missing %."""
    encoding = _load_encoding()
    target_desc = {spec["col"]: spec["desc"] for spec in TARGETS.values()}
    index_desc = {
        "resp_ahi_total": "Apnea-hypopnea index (events/hour), total, no RERAs.",
        "resp_rdi_total": "Respiratory disturbance index (events/hour), total, with RERAs.",
    }
    # Not present in dataset/encoding.json; inferred from the column name and
    # its binary 0/1 values (same missingness pattern as the sibling
    # mh_tobacco_habit column). Flagged as inferred, not curator-verified.
    fallback_definitions = {
        "mh_tobacco_history": "History of tobacco use (inferred from column name/values, not in the source data dictionary).",
    }

    rows = []
    for col in shared.columns:
        if col == "subject_id":
            definition = "Synthetic sequential subject identifier assigned on export; carries no information about collection order, site or the original record id."
            enc_str = ""
        elif col in target_desc:
            definition = f"Model target: {target_desc[col]}."
            if col.startswith("osa_four"):
                enc_str = "; ".join(f"{i}={name}" for i, name in enumerate(CLASS_NAMES))
            else:
                enc_str = "0=below threshold; 1=at or above threshold"
        elif col in index_desc:
            definition = index_desc[col] + " Continuous index used to derive the binary/four-class targets above."
            enc_str = ""
        elif col in fallback_definitions:
            definition = fallback_definitions[col]
            enc_str = "1=yes; 0=no (inferred)"
        else:
            meta = encoding.get(col, {})
            definition = meta.get("definition", "")
            enc = meta.get("encoding", {})
            enc_str = "; ".join(f"{v}={k}" for k, v in enc.items()) if enc else ""
            if col == "dem_age":
                enc_str = f"{AGE_TOP_CODE}={AGE_TOP_CODE} years or older (top-coded for HIPAA Safe Harbor)"

        rows.append(
            {
                "column": col,
                "definition": definition,
                "encoding": enc_str,
                "dtype": str(shared[col].dtype),
                "n_missing": int(shared[col].isna().sum()),
                "pct_missing": round(100 * shared[col].isna().mean(), 2),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    PATHS["shared_data_dir"].mkdir(parents=True, exist_ok=True)

    shared = build_shared_dataframe()
    shared.to_csv(PATHS["shared_data_csv"], index=False)

    data_dict = build_data_dictionary(shared)
    data_dict.to_csv(PATHS["shared_data_dictionary"], index=False)

    print(f"Wrote {len(shared):,} rows x {shared.shape[1]} columns -> {PATHS['shared_data_csv']}")
    print(f"Wrote data dictionary ({len(data_dict)} columns) -> {PATHS['shared_data_dictionary']}")


if __name__ == "__main__":
    main()
