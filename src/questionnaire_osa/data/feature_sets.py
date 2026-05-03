"""Construct the three feature configurations evaluated in the manuscript.

Definitions
-----------
* ``all_features``  – every column whose name starts with one of the
  prefixes in :data:`config.FEATURE_PREFIXES`.
* ``dem_only``      – ``dem_*`` columns only (≈ 4–5 demographic features).
* ``no_dem``        – ``all_features`` minus ``dem_*`` and ``bmi_*``
  (the *Questionnaire-Only* set).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import pandas as pd


def _select_by_prefix(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    """Return *columns* whose name starts with any of *prefixes*."""
    return [c for c in columns if any(c.startswith(p) for p in prefixes)]


def build_feature_sets(
    df: pd.DataFrame, feature_prefixes: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Construct the three feature-set dictionaries used by the trainers.

    Parameters
    ----------
    df
        Full subject-level dataframe (as returned by
        :func:`data.loader.load_sleep_data`).
    feature_prefixes
        Prefixes that delimit the feature universe — typically
        :data:`config.FEATURE_PREFIXES`.

    Returns
    -------
    dict
        ``{set_name: {"desc": str, "cols": list[str]}}``.
    """
    all_cols = _select_by_prefix(df.columns, feature_prefixes)

    return {
        "all_features": {
            "desc": "All Features",
            "cols": all_cols,
        },
        "dem_only": {
            "desc": "Demographics Only",
            "cols": [
                c for c in all_cols
                if c.startswith("dem_") and not c.startswith("dem_age_women_50")
            ],
        },
        "no_dem": {
            "desc": "Questionnaire Only",
            "cols": [
                c for c in all_cols
                if not c.startswith("dem_") and not c.startswith("bmi_")
            ],
        },
    }
