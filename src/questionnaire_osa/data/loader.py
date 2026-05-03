"""SQLite loader for the cleaned sleep-research database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

TABLE_NAME: str = "sleep_data"


def load_sleep_data(db_path: Path, min_age: int = 18) -> pd.DataFrame:
    """Read the *adults-only* slice of the sleep-research database.

    Parameters
    ----------
    db_path
        Absolute path to ``sleep_research_cleaned.db``.
    min_age
        Inclusive lower bound on ``dem_age``.  Defaults to 18 to match
        the manuscript inclusion criterion.

    Returns
    -------
    pandas.DataFrame
        Single dataframe with one row per subject and 270 columns.

    Raises
    ------
    FileNotFoundError
        If ``db_path`` does not exist.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}")

    query = f"SELECT * FROM {TABLE_NAME} WHERE dem_age >= {int(min_age)}"
    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn)
    return df
