"""Build the cohort descriptive table (Supplementary Table S3).

The table stratifies the adult cohort by AHI-based OSA severity
(Normal / Mild / Moderate / Severe, from ``osa_four_numeric``) and reports
demographics, anthropometrics and the key sleep-symptom items.

Every function here is pure: it takes a dataframe and returns a new one.
Percentages for categorical items are computed over non-missing values
within each column; missing counts are reported on their own row.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Table layout ──────────────────────────────────────────────────────────────
SEVERITY_COL: str = "osa_four_numeric"
SEVERITY_LABELS: Dict[int, str] = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}

HEADER: List[str] = ["Variable", "Category / Statistic"]
OBESITY_BMI_CUTOFF: float = 30.0

# Likert items shown with their full category breakdown, in table order.
LIKERT_ITEMS: List[Tuple[str, str]] = [
    ("sa_sleep_snore",                 "Snoring"),
    ("sa_sleep_stop_breath",           "Witnessed Apneas — Stop Breathing"),
    ("sa_sleep_gasp",                  "Witnessed Apneas — Gasping"),
    ("sa_sleep_restless",              "Sleep Restless"),
    ("sa_problem_going_tosleep_atnight", "Problem Going to Sleep at Night"),
    ("sa_sleep_grind_teeth",           "Grind Teeth"),
]
LIKERT_LEVELS: List[Tuple[int, str]] = [
    (0, "Not at all / None"),
    (1, "Slight / Few times"),
    (2, "Moderate / Sometimes"),
    (3, "Often"),
    (4, "Severe / Always"),
]

# Binary yes/no items shown as a single "n (% non-missing)" row.
BINARY_ITEMS: List[Tuple[str, str]] = [
    ("mh_high_blood_pressure",   "High Blood Pressure"),
    ("sa_before_bed_sleep_aids", "Before Bed Sleep Aid"),
]

FOOTNOTE: str = (
    "Notes: Percentages computed over non-missing values within each column. "
    "Ordinal scale (0–4): 0 = Not at all / None, 1 = Slight / Few times, "
    "2 = Moderate / Sometimes, 3 = Often, 4 = Severe / Always. "
    "Obese: BMI ≥ 30 kg/m². Female Age > 50: female subjects aged > 50 years. "
    "Severity groups defined on AHI: Normal <5, Mild 5–<15, Moderate 15–<30, "
    "Severe ≥30 events/h. Cohort restricted to adults aged ≥ 18 years "
    "(inclusive), identical to the model training cohort. "
    "One observation per unique subject id."
)


# ── Formatting helpers ────────────────────────────────────────────────────────
def _fmt_n_pct(n: int, denom: int) -> str:
    pct = 100.0 * n / denom if denom else 0.0
    return f"{n:,} ({pct:.1f}%)"


def _fmt_mean_sd(s: pd.Series) -> str:
    return f"{s.mean():.1f} ± {s.std():.1f}"


def _fmt_median_iqr(s: pd.Series) -> str:
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{med:.1f} [{q1:.1f}–{q3:.1f}]"


def _decode_labels(encoding: Mapping[str, object], col: str) -> Dict[int, str]:
    """Return ``{code: Title-cased label}`` for an encoded categorical column."""
    entry = encoding.get(col, {})
    codes = entry.get("encoding", {}) if isinstance(entry, dict) else {}
    return {int(v): _title(k) for k, v in codes.items()}


def _title(label: str) -> str:
    # "white (caucasian)" → "White (Caucasian)"
    return " ".join(
        w if w.startswith("(") else w.capitalize()
        for w in label.split(" ")
    ).replace("(c", "(C").replace("(h", "(H")


# ── Row builders (each returns a list of rows; a row is a list of cells) ──────
def _continuous_rows(label: str, groups: Sequence[pd.DataFrame], col: str) -> List[list]:
    return [
        [label, None, *[None] * len(groups)],
        [None, "  Mean ± SD",    *[_fmt_mean_sd(g[col].dropna())    for g in groups]],
        [None, "  Median [IQR]", *[_fmt_median_iqr(g[col].dropna()) for g in groups]],
    ]


def _categorical_rows(
    label: str,
    groups: Sequence[pd.DataFrame],
    col: str,
    levels: Sequence[Tuple[int, str]],
    show_missing: bool,
) -> List[list]:
    rows: List[list] = [[label, None, *[None] * len(groups)]]
    non_missing = [g[col].notna().sum() for g in groups]
    for code, level_label in levels:
        rows.append([
            None, f"  {level_label}",
            *[_fmt_n_pct(int((g[col] == code).sum()), int(nm))
              for g, nm in zip(groups, non_missing)],
        ])
    if show_missing:
        rows.append([None, "  Missing (n)", *[int(g[col].isna().sum()) for g in groups]])
    return rows


def _binary_row(label: str, groups: Sequence[pd.DataFrame], mask_fn) -> list:
    cells = []
    for g in groups:
        mask = mask_fn(g)
        cells.append(_fmt_n_pct(int(mask.sum()), int(mask.notna().sum())))
    return [label, "n (% non-missing)", *cells]


# ── Public API ────────────────────────────────────────────────────────────────
def build_table_one(df: pd.DataFrame, encoding: Mapping[str, object]) -> pd.DataFrame:
    """Return the Table S3 dataframe (header row included as row 0).

    Parameters
    ----------
    df
        Adult cohort as returned by :func:`data.loader.load_sleep_data`.
    encoding
        Parsed ``encoding.json`` (``{column: {"definition", "encoding"}}``).
    """
    if SEVERITY_COL not in df.columns:
        raise KeyError(f"Severity column '{SEVERITY_COL}' missing from dataframe")

    groups: List[pd.DataFrame] = [df] + [
        df[df[SEVERITY_COL] == code] for code in SEVERITY_LABELS
    ]
    group_names = ["Overall"] + list(SEVERITY_LABELS.values())
    header = HEADER + [f"{name} (N={len(g):,})" for name, g in zip(group_names, groups)]

    gender = sorted(_decode_labels(encoding, "dem_gender").items())
    race = sorted(_decode_labels(encoding, "dem_race").items())

    rows: List[list] = [header]
    rows += _continuous_rows("Age (years)", groups, "dem_age")
    rows += _categorical_rows("Gender", groups, "dem_gender", gender, show_missing=False)
    rows += _categorical_rows("Race / Ethnicity", groups, "dem_race", race, show_missing=False)
    rows += _continuous_rows("BMI (kg/m²)", groups, "dem_bmi")
    rows.append(_binary_row(
        f"Obese (BMI ≥ {OBESITY_BMI_CUTOFF:.0f})", groups,
        lambda g: (g["dem_bmi"] >= OBESITY_BMI_CUTOFF).where(g["dem_bmi"].notna()),
    ))
    rows.append(_binary_row(
        "Female Age > 50", groups,
        lambda g: (g["dem_age_women_50"] == 1).where(g["dem_age_women_50"].notna()),
    ))
    for col, label in LIKERT_ITEMS:
        rows += _categorical_rows(label, groups, col, LIKERT_LEVELS, show_missing=True)
    for col, label in BINARY_ITEMS:
        rows.append(_binary_row(
            label, groups, lambda g, c=col: (g[c] == 1).where(g[c].notna())
        ))
    rows.append([None] * len(header))
    rows.append([FOOTNOTE, *[None] * (len(header) - 1)])

    return pd.DataFrame(rows).replace({np.nan: None})


def write_table_one(table: pd.DataFrame, out_path, sheet_name: str = "Table S3 — OSA Severity") -> None:
    """Write *table* to an .xlsx file with no pandas header/index."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        table.to_excel(xw, sheet_name=sheet_name, header=False, index=False)
        ws = xw.sheets[sheet_name]
        ws.column_dimensions["A"].width = 38
        ws.column_dimensions["B"].width = 26
        for col_letter in "CDEFG":
            ws.column_dimensions[col_letter].width = 22
