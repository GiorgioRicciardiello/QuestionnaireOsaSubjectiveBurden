"""Entry point: regenerate every figure & table from a stored XGBoost run.

Run with::

    python -m questionnaire_osa.analyze_xgboost

Reads :data:`config.PATHS["xgb_results_pkl"]` and writes:

* ``figures/performance/roc_curves.png``
* ``figures/performance/confusion_matrices_binary.png``
* ``figures/performance/confusion_matrices_multiclass.png``
* ``figures/shap/shap_beeswarm_binary.png``
* ``figures/shap/shap_beeswarm_binary_no_dem.png``
* ``figures/shap/shap_beeswarm_<ahi|rdi>_4class.png``
* ``figures/shap_strata/shap_beeswarm_<age|bmi>_<target>.png``
* ``figures/shap_dependence/shap_dependence_3x3_<target>.png``
* ``tables/shap_feature_importance_binary.csv``
* ``tables/shap_feature_importance_multiclass.csv``
* ``tables/results_summary.csv``
"""

from __future__ import annotations

import pickle
from typing import Any, Dict

from config import (
    PATHS, SETTINGS, AGE_GROUPS, BMI_GROUPS, CLASS_NAMES, FS_COLORS,
    FS_ORDER, SHAP_MAX_DISPLAY, SHAP_SAMPLE_SIZE,
)
from config.config import ensure_output_dirs

from .plots import (
    apply_publication_style,
    plot_confusion_matrices_binary,
    plot_confusion_matrices_multiclass,
    plot_roc_curves,
)
from .reporting import build_results_summary_csv, print_results_summary
from .shap_analysis import (
    compute_shap_values,
    export_importance_csv,
    get_model_and_test,
    plot_shap_beeswarm_binary,
    plot_shap_beeswarm_multiclass,
    plot_shap_dependence_3x3,
    plot_shap_strata_beeswarm,
    rename_features,
    shap_by_strata,
)
from .utils import set_global_seeds


PRIMARY_FS = "all_features"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load_artifacts(path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Training artifacts not found at {path}\n"
            "Run `python -m questionnaire_osa.train_xgboost` first."
        )
    with path.open("rb") as fh:
        return pickle.load(fh)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    set_global_seeds(SETTINGS["random_state"])
    ensure_output_dirs()
    apply_publication_style()

    artifacts          = _load_artifacts(PATHS["xgb_results_pkl"])
    feature_sets       = artifacts["feature_sets"]
    targets_config     = artifacts["targets_config"]
    binary_targets     = artifacts["binary_targets"]
    multiclass_targets = artifacts["multiclass_targets"]
    all_targets        = artifacts["all_targets"]
    display_names      = artifacts["feature_display_names"]
    random_state       = artifacts["settings"]["random_state"]

    print(f"Loaded {len(feature_sets)} feature sets, {artifacts['n_samples']:,} samples")
    print_results_summary(feature_sets)

    # ── Long-form CSV for the manuscript ──────────────────────────────────────
    build_results_summary_csv(feature_sets, PATHS["tbl_results_summary"])
    print(f"\nResults CSV: {PATHS['tbl_results_summary']}")

    # ── Performance plots ─────────────────────────────────────────────────────
    plot_roc_curves(
        feature_sets, binary_targets, targets_config,
        FS_ORDER, FS_COLORS,
        save_path=PATHS["fig_performance"] / "roc_curves.png",
    )
    plot_confusion_matrices_binary(
        feature_sets[PRIMARY_FS]["all_results"],
        feature_sets[PRIMARY_FS]["all_test_sets"],
        binary_targets, targets_config,
        save_path=PATHS["fig_performance"] / "confusion_matrices_binary.png",
    )
    plot_confusion_matrices_multiclass(
        feature_sets[PRIMARY_FS]["all_results"],
        feature_sets[PRIMARY_FS]["all_test_sets"],
        multiclass_targets, targets_config,
        save_path=PATHS["fig_performance"] / "confusion_matrices_multiclass.png",
    )
    print(f"Performance figures saved → {PATHS['fig_performance']}")

    # ── SHAP: beeswarms (sampled) ─────────────────────────────────────────────
    print("\nComputing SHAP values (All-Features model)...")
    all_shap = compute_shap_values(
        feature_sets, "all_features", all_targets, targets_config,
        display_names, sample_size=SHAP_SAMPLE_SIZE, random_state=random_state,
    )

    plot_shap_beeswarm_binary(
        all_shap, binary_targets, targets_config,
        suptitle=("SHAP-Based Feature Attribution in a Binary-Class\n"
                  "OSA Severity Classifier Using All Features"),
        save_path=PATHS["fig_shap"] / "shap_beeswarm_binary.png",
        max_display=SHAP_MAX_DISPLAY,
    )

    print("Computing SHAP values (Questionnaire-Only model)...")
    no_dem_shap = compute_shap_values(
        feature_sets, "no_dem", binary_targets, targets_config,
        display_names, sample_size=SHAP_SAMPLE_SIZE, random_state=random_state,
    )
    plot_shap_beeswarm_binary(
        no_dem_shap, binary_targets, targets_config,
        suptitle=("SHAP-Based Feature Attribution in a Binary-Class\n"
                  "OSA Severity Classifier Using Sleep Questionnaire Data"),
        save_path=PATHS["fig_shap"] / "shap_beeswarm_binary_no_dem.png",
        max_display=SHAP_MAX_DISPLAY,
    )

    # Multiclass beeswarms (one figure per 4-class target)
    multiclass_titles = {
        "ahi_4class": ("SHAP-Based Feature Attribution in a Four-Class\n"
                       "OSA Severity Classifier Using AHI and Sleep Questionnaire Data"),
        "rdi_4class": ("SHAP-Based Feature Attribution in a Four-Class\n"
                       "OSA Severity Classifier Using RDI and Sleep Questionnaire Data"),
    }
    for target in multiclass_targets:
        plot_shap_beeswarm_multiclass(
            all_shap, target, CLASS_NAMES,
            suptitle=multiclass_titles[target],
            save_path=PATHS["fig_shap"] / f"shap_beeswarm_{target}.png",
            max_display=SHAP_MAX_DISPLAY,
        )
    print(f"SHAP beeswarms saved → {PATHS['fig_shap']}")

    # ── SHAP: feature-importance CSVs (full test set, no sampling) ────────────
    print("\nComputing full-test-set SHAP importances...")
    csv_binary_per_fs = [
        ("All Features",       compute_shap_values(
            feature_sets, "all_features", binary_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
        ("Questionnaire Only", compute_shap_values(
            feature_sets, "no_dem", binary_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
        ("Demographics Only",  compute_shap_values(
            feature_sets, "dem_only", binary_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
    ]
    export_importance_csv(
        csv_binary_per_fs, binary_targets, targets_config,
        output_path=PATHS["tbl_shap_binary"],
    )
    print(f"Binary CSV: {PATHS['tbl_shap_binary']}")

    csv_mc_per_fs = [
        ("All Features",       compute_shap_values(
            feature_sets, "all_features", multiclass_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
        ("Questionnaire Only", compute_shap_values(
            feature_sets, "no_dem", multiclass_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
        ("Demographics Only",  compute_shap_values(
            feature_sets, "dem_only", multiclass_targets, targets_config,
            display_names, sample_size=None, random_state=random_state)),
    ]
    export_importance_csv(
        csv_mc_per_fs, multiclass_targets, targets_config,
        output_path=PATHS["tbl_shap_multiclass"],
        drop_columns=["Std", "Direction"],
    )
    print(f"Multiclass CSV: {PATHS['tbl_shap_multiclass']}")

    # ── SHAP: demographic strata + dependence plot ────────────────────────────
    # Re-compute SHAP on the *full* test set for each binary AHI target —
    # necessary for stratum / dependence analysis (sampled values are too sparse).
    print("\nRecomputing SHAP on full test sets (binary AHI targets)...")
    full_shap: Dict[str, Dict[str, Any]] = {}
    import shap as shap_lib

    for target in binary_targets:
        model, X_test, y_test = get_model_and_test(feature_sets, "all_features", target)
        explainer = shap_lib.TreeExplainer(model)
        full_shap[target] = {
            "shap_values": explainer.shap_values(X_test),
            "X_test":      rename_features(X_test, display_names),
            "y_test":      y_test,
        }
        print(f"  [{target}] {full_shap[target]['shap_values'].shape}")

    # Dependence plot for AHI ≥ 5 (primary clinical threshold)
    plot_shap_dependence_3x3(
        full_shap["ahi_5"]["shap_values"],
        full_shap["ahi_5"]["X_test"],
        save_path=PATHS["fig_shap_dependence"] / "shap_dependence_3x3_ahi_5.png",
        target_desc=f"in the OSA Severity Classifier ({targets_config['ahi_5']['desc'].replace(' @ ', ' ≥ ')})",
        random_state=random_state,
    )
    print(f"SHAP dependence: {PATHS['fig_shap_dependence']}")

    # Stratum analyses (Age then BMI) for the three AHI binary targets
    ahi_targets = ["ahi_5", "ahi_15", "ahi_30"]
    for target in ahi_targets:
        for stratum_label, stratum_col, groups in (
            ("Age", "Age", AGE_GROUPS),
            ("BMI", "BMI", BMI_GROUPS),
        ):
            try:
                groups_data = shap_by_strata(
                    full_shap[target]["shap_values"],
                    full_shap[target]["X_test"],
                    stratum_col, groups,
                )
            except KeyError:
                print(f"  Skipping {stratum_label} stratum: '{stratum_col}' "
                      "not present after rename.")
                continue
            desc = targets_config[target]["desc"].replace(" @ ", " ≥ ")
            plot_shap_strata_beeswarm(
                groups_data,
                suptitle=f"SHAP Beeswarm by {stratum_label} Group — {desc}",
                save_path=(
                    PATHS["fig_shap_strata"]
                    / f"shap_beeswarm_{stratum_label.lower()}_{target}.png"
                ),
                max_display=SHAP_MAX_DISPLAY,
            )
    print(f"Stratum figures saved → {PATHS['fig_shap_strata']}")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
