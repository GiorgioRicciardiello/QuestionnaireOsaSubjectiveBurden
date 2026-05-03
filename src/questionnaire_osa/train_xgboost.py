"""Entry point: train XGBoost models for all (target × feature-set) cells.

Run with::

    python -m questionnaire_osa.train_xgboost

No CLI arguments — every knob lives in :mod:`config.config`.

Output
------
* :data:`config.PATHS["xgb_results_pkl"]` — single pickle containing
  models, hyper-parameters, predictions, test sets, and the full source
  dataframe needed by the analysis entry points.
"""

from __future__ import annotations

import pickle
from datetime import datetime

import optuna
import xgboost as xgb

from config import (
    PATHS, SETTINGS, TARGETS, BINARY_TARGETS, MULTICLASS_TARGETS,
    ALL_TARGETS, FEATURE_PREFIXES, STRATIFICATION_CONFIG,
)
from config.config import ensure_output_dirs

from .data import build_feature_sets, load_sleep_data
from .models import train_all_targets_xgb
from .utils import load_feature_display_names, set_global_seeds


def main() -> None:
    set_global_seeds(SETTINGS["random_state"])
    ensure_output_dirs()

    print(f"XGBoost: {xgb.__version__}, Optuna: {optuna.__version__}")
    print(f"Random seed: {SETTINGS['random_state']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nLoading data from {PATHS['database']}")
    df = load_sleep_data(PATHS["database"], min_age=SETTINGS["min_age"])
    print(f"  Loaded: {len(df):,} samples × {len(df.columns)} columns")

    feature_sets = build_feature_sets(df, FEATURE_PREFIXES)
    for name, cfg in feature_sets.items():
        print(f"  {name}: {len(cfg['cols'])} features ({cfg['desc']})")

    display_names = load_feature_display_names(PATHS["encoding_json"])
    print(f"  Display names loaded for {len(display_names)} features")

    # ── Train ─────────────────────────────────────────────────────────────────
    feature_set_results: dict = {}
    for fs_name, fs_cfg in feature_sets.items():
        X_fs = df[fs_cfg["cols"]]
        print(f"\n{'#' * 70}")
        print(f"FEATURE SET: {fs_cfg['desc']} ({len(fs_cfg['cols'])} features)")
        print("#" * 70)

        feature_set_results[fs_name] = train_all_targets_xgb(
            X_fs, df, TARGETS, STRATIFICATION_CONFIG,
            train_pct=SETTINGS["train_pct"],
            val_pct=SETTINGS["val_pct"],
            classification_threshold=SETTINGS["classification_threshold"],
            n_trials=SETTINGS["n_trials"],
            n_folds=SETTINGS["n_folds"],
            random_state=SETTINGS["random_state"],
        )

    # ── Persist ───────────────────────────────────────────────────────────────
    artifacts = {
        "feature_sets": {
            fs_name: {
                **fs_results,
                "feature_cols": feature_sets[fs_name]["cols"],
                "n_features":   len(feature_sets[fs_name]["cols"]),
                "desc":         feature_sets[fs_name]["desc"],
            }
            for fs_name, fs_results in feature_set_results.items()
        },
        "df":                     df,
        "X_all":                  df[feature_sets["all_features"]["cols"]],
        "targets_config":         TARGETS,
        "stratification_config":  STRATIFICATION_CONFIG,
        "feature_display_names":  display_names,
        "feature_set_names":      list(feature_sets.keys()),
        "feature_sets_config":    feature_sets,
        "settings":               SETTINGS,
        "binary_targets":         BINARY_TARGETS,
        "multiclass_targets":     MULTICLASS_TARGETS,
        "all_targets":            ALL_TARGETS,
        "n_samples":              len(df),
        "metadata": {
            "created_at":      datetime.now().isoformat(),
            "pipeline_version": "1.0.0",
        },
    }

    out_path = PATHS["xgb_results_pkl"]
    if out_path.exists():
        backup = out_path.with_name(
            f"{out_path.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{out_path.suffix}"
        )
        out_path.rename(backup)
        print(f"\nBackup created: {backup.name}")

    with out_path.open("wb") as fh:
        pickle.dump(artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    total_models = sum(
        len(r["all_models"]) for r in feature_set_results.values()
    )
    print(f"\nSaved: {out_path}  ({size_mb:.2f} MB)")
    print(f"Total models: {total_models} ({len(feature_sets)} feature sets × {len(TARGETS)} targets)")


if __name__ == "__main__":
    main()
