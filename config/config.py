"""Single source of truth for filesystem paths and run-time constants.

No path string is allowed inside :mod:`questionnaire_osa`; every module
that needs a path must import from :data:`PATHS`.

Layout (resolved at import time):

    <project_root>/
    ├── config/
    ├── dataset/
    ├── results/      ← .pkl model + CV artifacts
    ├── figures/      ← .png plots (organised in sub-folders by analysis)
    ├── tables/       ← .csv summary tables
    └── src/questionnaire_osa/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ── Project root resolution ───────────────────────────────────────────────────
# config/config.py  →  config/  →  project root
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Single immutable mapping that every script imports from.
PATHS: Dict[str, Path] = {
    # Top-level directories
    "project_root":       PROJECT_ROOT,
    "config_dir":         PROJECT_ROOT / "config",
    "dataset_dir":        PROJECT_ROOT / "dataset",
    "results_dir":        PROJECT_ROOT / "results",
    "figures_dir":        PROJECT_ROOT / "figures",
    "tables_dir":         PROJECT_ROOT / "tables",
    "src_dir":            PROJECT_ROOT / "src" / "questionnaire_osa",

    # Input artefacts
    "database":           PROJECT_ROOT / "dataset" / "sleep_research_cleaned.db",
    "encoding_json":      PROJECT_ROOT / "dataset" / "encoding.json",

    # Output: pickle artefacts (one per training run)
    "xgb_results_pkl":    PROJECT_ROOT / "results" / "xgboost_cv_results.pkl",
    "tabpfn_results_pkl": PROJECT_ROOT / "results" / "tabpfn_cv_results.pkl",

    # Output: figure sub-directories
    "fig_performance":    PROJECT_ROOT / "figures" / "performance",
    "fig_shap":           PROJECT_ROOT / "figures" / "shap",
    "fig_shap_strata":    PROJECT_ROOT / "figures" / "shap_strata",
    "fig_shap_dependence":PROJECT_ROOT / "figures" / "shap_dependence",
    "fig_stratification": PROJECT_ROOT / "figures" / "stratification",
    "fig_tabpfn":         PROJECT_ROOT / "figures" / "tabpfn",

    # Output: tables
    "tbl_shap_binary":     PROJECT_ROOT / "tables" / "shap_feature_importance_binary.csv",
    "tbl_shap_multiclass": PROJECT_ROOT / "tables" / "shap_feature_importance_multiclass.csv",
    "tbl_results_summary": PROJECT_ROOT / "tables" / "results_summary.csv",
}


def ensure_output_dirs() -> None:
    """Create every output directory on disk (idempotent).

    Should be called once at the start of any entry-point script that
    writes artefacts.
    """
    for key in (
        "results_dir", "figures_dir", "tables_dir",
        "fig_performance", "fig_shap", "fig_shap_strata",
        "fig_shap_dependence", "fig_stratification", "fig_tabpfn",
    ):
        PATHS[key].mkdir(parents=True, exist_ok=True)


# ── Run-time settings ─────────────────────────────────────────────────────────
SETTINGS: Dict[str, Any] = {
    "random_state":              42,
    "n_trials":                  100,    # Optuna trials per (target, feature-set)
    "n_folds":                   5,      # CV folds for both XGBoost & TabPFN
    "train_pct":                 90,     # 90/10 train_val/test split
    "val_pct":                   10,
    "classification_threshold":  0.5,
    "min_age":                   18,     # adults only
    "tabpfn_device":             "cuda", # set to "cpu" if no GPU available
    "tabpfn_token_env":          "TABPFN_TOKEN",
}


# ── Feature-set definitions ───────────────────────────────────────────────────
# Column-name prefixes that delimit the feature universe.
FEATURE_PREFIXES: List[str] = [
    "presleep_", "postsleep_", "sa_", "mh_", "ep_", "dem_", "bmi_",
]

# Names of the three feature configurations evaluated in the manuscript.
# The actual column lists are derived at runtime in :mod:`data.feature_sets`.
FEATURE_SET_NAMES: List[str] = ["all_features", "dem_only", "no_dem"]


# ── Targets ───────────────────────────────────────────────────────────────────
# 8 targets total: 6 binary thresholds (AHI/RDI × {5, 15, 30}) + 2 four-class.
TARGETS: Dict[str, Dict[str, str]] = {
    "ahi_5":      {"col": "osa_binary_numeric",                 "type": "binary",     "group": "AHI", "desc": "AHI @ 5"},
    "ahi_15":     {"col": "osa_binary_fifteenth_numeric",       "type": "binary",     "group": "AHI", "desc": "AHI @ 15"},
    "ahi_30":     {"col": "osa_binary_thirty_numeric",          "type": "binary",     "group": "AHI", "desc": "AHI @ 30"},
    "rdi_5":      {"col": "osa_binary_numeric_rdi",             "type": "binary",     "group": "RDI", "desc": "RDI @ 5"},
    "rdi_15":     {"col": "osa_binary_fifteenth_numeric_rdi",   "type": "binary",     "group": "RDI", "desc": "RDI @ 15"},
    "rdi_30":     {"col": "osa_binary_thirty_numeric_rdi",      "type": "binary",     "group": "RDI", "desc": "RDI @ 30"},
    "ahi_4class": {"col": "osa_four_numeric",                   "type": "multiclass", "group": "AHI", "desc": "AHI 4-class"},
    "rdi_4class": {"col": "osa_four_numeric_rdi",               "type": "multiclass", "group": "RDI", "desc": "RDI 4-class"},
}

BINARY_TARGETS:     List[str] = [k for k, v in TARGETS.items() if v["type"] == "binary"]
MULTICLASS_TARGETS: List[str] = [k for k, v in TARGETS.items() if v["type"] == "multiclass"]
ALL_TARGETS:        List[str] = list(TARGETS.keys())


# ── Stratification configuration ──────────────────────────────────────────────
# Variables used jointly with the target to construct the multilabel
# stratification matrix (see :mod:`stratification.splitter`).
STRATIFICATION_CONFIG: Dict[str, Dict[str, Any]] = {
    "dem_age":              {"type": "continuous",  "bins": [0, 30, 40, 50, 60, 70, 100]},
    "dem_gender":           {"type": "categorical"},
    "dem_bmi":              {"type": "continuous",  "bins": [0, 18.5, 25, 30, 35, 40, 100]},
    "dem_race":             {"type": "categorical"},
    "sa_sleep_snore":       {"type": "categorical"},
    "sa_sleep_stop_breath": {"type": "categorical"},
    "LAB_CAT":              {"type": "categorical"},
    "MD_Identifyer":        {"type": "categorical"},
}


# ── SHAP analysis strata ──────────────────────────────────────────────────────
# Subgroup definitions used in the demographic SHAP analysis.
AGE_GROUPS: Dict[str, tuple] = {
    "<=36":  (0,  36),
    "36-46": (36, 46),
    "46-58": (46, 58),
    ">=58":  (58, 100),
}

BMI_GROUPS: Dict[str, tuple] = {
    "<=26":  (0,  26),
    "26-30": (26, 30),
    "30-34": (30, 34),
    ">=34":  (34, 100),
}


# ── Plotting & display ────────────────────────────────────────────────────────
CLASS_NAMES: List[str] = ["No OSA", "Mild OSA", "Moderate", "Severe"]

# Display labels and colors for the 3 feature-set comparison plots.
# Colors follow the Okabe-Ito palette (color-blind safe).
FS_LABELS: Dict[str, str] = {
    "dem_only":     "Demographics\nOnly",
    "no_dem":       "Questionnaire\nOnly",
    "all_features": "All\nFeatures",
}

FS_COLORS: Dict[str, str] = {
    "dem_only":     "#0072B2",   # blue
    "no_dem":       "#D55E00",   # vermillion
    "all_features": "#009E73",   # bluish green
}

FS_ORDER: List[str] = ["dem_only", "no_dem", "all_features"]


# ── SHAP analysis tuning ──────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE: int = 1000   # subsample size for beeswarm plots (None = full set)
SHAP_MAX_DISPLAY: int = 10     # top-N features displayed in beeswarm plots
