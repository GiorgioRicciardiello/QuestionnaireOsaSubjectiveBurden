"""Entry point: cross-validate TabPFN v2.5 across all 8 targets.

Run with::

    python -m questionnaire_osa.train_tabpfn

No CLI arguments.  Authentication: the TabPFN client requires a
``TABPFN_TOKEN`` environment variable; the variable name is itself
configured in :data:`config.SETTINGS["tabpfn_token_env"]`.

Output
------
:data:`config.PATHS["tabpfn_results_pkl"]` — fold-level CV artefacts.
"""

from __future__ import annotations

import os
import pickle
import time
from datetime import datetime

from config import (
    PATHS, SETTINGS, TARGETS, BINARY_TARGETS, MULTICLASS_TARGETS,
    ALL_TARGETS, FEATURE_PREFIXES, STRATIFICATION_CONFIG,
)
from config.config import ensure_output_dirs

from .data import build_feature_sets, load_sleep_data
from .models import run_all_targets_tabpfn
from .utils import set_global_seeds


def main() -> None:
    set_global_seeds(SETTINGS["random_state"])
    ensure_output_dirs()

    token_env = SETTINGS["tabpfn_token_env"]
    if token_env not in os.environ:
        print(
            f"WARNING: ${token_env} is not set — TabPFN inference will fail.\n"
            f"         Run `setx {token_env} <your-token>` (Windows) or\n"
            f"         export {token_env}=<your-token>` (Unix) before launching."
        )

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"Loading data from {PATHS['database']}")
    df = load_sleep_data(PATHS["database"], min_age=SETTINGS["min_age"])
    print(f"  Loaded: {len(df):,} samples × {len(df.columns)} columns")

    feature_sets = build_feature_sets(df, FEATURE_PREFIXES)
    X_all_cols = feature_sets["all_features"]["cols"]
    X = df[X_all_cols]
    print(f"  Feature columns: {len(X_all_cols)}")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    cv_artifacts = run_all_targets_tabpfn(
        X, df, TARGETS, STRATIFICATION_CONFIG,
        n_splits=SETTINGS["n_folds"],
        classification_threshold=SETTINGS["classification_threshold"],
        device=SETTINGS["tabpfn_device"],
        random_state=SETTINGS["random_state"],
    )
    print(f"\nAll experiments complete ({(time.time() - t0) / 60:.1f} min)")

    # ── Persist ───────────────────────────────────────────────────────────────
    artifacts = {
        **cv_artifacts,
        "targets_config":        TARGETS,
        "stratification_config": STRATIFICATION_CONFIG,
        "feature_cols":          X_all_cols,
        "df":                    df,
        "X":                     X,
        "n_samples":             len(df),
        "n_features":            len(X_all_cols),
        "n_folds":               SETTINGS["n_folds"],
        "binary_targets":        BINARY_TARGETS,
        "multiclass_targets":    MULTICLASS_TARGETS,
        "all_targets":           ALL_TARGETS,
        "settings":              SETTINGS,
        "metadata": {
            "created_at":       datetime.now().isoformat(),
            "model":            "TabPFN v2.5",
            "pipeline_version": "1.0.0",
        },
    }

    out_path = PATHS["tabpfn_results_pkl"]
    with out_path.open("wb") as fh:
        pickle.dump(artifacts, fh, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
