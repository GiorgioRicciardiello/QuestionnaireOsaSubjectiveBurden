"""Entry point: regenerate every figure & table from a stored XGBoost run.

Run with::

    python -m questionnaire_osa.analyze_xgboost

Reads :data:`config.PATHS["xgb_results_pkl"]` and writes final, manuscript-
numbered figures/tables straight to their task-based output directories
(:data:`config.PATHS["fig_model_performance"]`, ``fig_feature_importance``,
``fig_feature_importance_stratified``, ``fig_feature_interactions``), with
everything NOT referenced by the manuscript or supplementary docx routed to
the gitignored draft archive (:data:`config.PATHS["fig_draft"]`):

* ``results/figures/model_performance/figureS4_confusion_multiclass.png``
* ``results/figures/feature_importance/figure3_shap_binary.png``
* ``results/figures/feature_importance/figureS5_shap_4class_ahi.png``
* ``results/figures/feature_importance/figureS6_shap_4class_rdi.png``
* ``results/figures/feature_importance_stratified/figureS11-S14_shap_<age|bmi>_ahi<15|30>.png``
* ``results/figures/feature_interactions/figure5_shap_dependence.png``
* ``results/figures/feature_interactions/figureS10_shap_dependence_extra.png``
* ``results/tables/feature_importance/shap_feature_importance_binary.csv``
* ``results/tables/feature_importance/shap_feature_importance_multiclass.csv``
* ``results/tables/feature_importance/shap_interaction_ranking_ahi_5.csv``
* ``results/tables/model_performance/results_summary.csv``

A number of exploratory / unused plots (plain ROC, binary confusion
matrices, un-stratified beeswarms, the ahi_5 strata panels superseded by
hand-assembled Figure 4) are also emitted on every run; those go to
:data:`config.PATHS["fig_draft"]`, which lives under the gitignored
``_local/`` archive.
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
    plot_shap_beeswarm_models,
    plot_shap_dependence_main,
    plot_shap_dependence_supplement,
    rank_interactions,
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
        save_path=PATHS["fig_draft"] / "roc_curves.png",  # unused; see figure1_roc (make_figure1.py)
    )
    plot_confusion_matrices_binary(
        feature_sets[PRIMARY_FS]["all_results"],
        feature_sets[PRIMARY_FS]["all_test_sets"],
        binary_targets, targets_config,
        save_path=PATHS["fig_draft"] / "confusion_matrices_binary.png",  # unused
    )
    plot_confusion_matrices_multiclass(
        feature_sets[PRIMARY_FS]["all_results"],
        feature_sets[PRIMARY_FS]["all_test_sets"],
        multiclass_targets, targets_config,
        save_path=PATHS["fig_model_performance"] / "figureS4_confusion_multiclass.png",
    )
    print(f"Performance figures saved → {PATHS['fig_model_performance']} (+ draft: {PATHS['fig_draft']})")

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
        save_path=PATHS["fig_draft"] / "shap_beeswarm_binary.png",  # unused; see figure3_shap_binary
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
        save_path=PATHS["fig_draft"] / "shap_beeswarm_binary_no_dem.png",  # unused
        max_display=SHAP_MAX_DISPLAY,
    )

    # Manuscript Figure 3: AHI thresholds, All-Features over Questionnaire-Only
    plot_shap_beeswarm_models(
        [("All Features", all_shap), ("Questionnaire Only", no_dem_shap)],
        ["ahi_5", "ahi_15", "ahi_30"], targets_config,
        suptitle="SHAP-Based Feature Importance in Binary-Class OSA Classification",
        save_path=PATHS["fig_feature_importance"] / "figure3_shap_binary.png",
        max_display=SHAP_MAX_DISPLAY,
    )

    # Multiclass beeswarms (one figure per 4-class target) → Supplementary Figures S5 (AHI), S6 (RDI)
    multiclass_titles = {
        "ahi_4class": ("SHAP-Based Feature Attribution in a Four-Class\n"
                       "OSA Severity Classifier Using AHI and Sleep Questionnaire Data"),
        "rdi_4class": ("SHAP-Based Feature Attribution in a Four-Class\n"
                       "OSA Severity Classifier Using RDI and Sleep Questionnaire Data"),
    }
    multiclass_filenames = {
        "ahi_4class": "figureS5_shap_4class_ahi.png",
        "rdi_4class": "figureS6_shap_4class_rdi.png",
    }
    for target in multiclass_targets:
        plot_shap_beeswarm_multiclass(
            all_shap, target, CLASS_NAMES,
            suptitle=multiclass_titles[target],
            save_path=PATHS["fig_feature_importance"] / multiclass_filenames[target],
            max_display=SHAP_MAX_DISPLAY,
        )
    print(f"SHAP beeswarms saved → {PATHS['fig_feature_importance']}")

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

    # Dependence figures for AHI ≥ 5 (primary clinical threshold):
    # strongest interactions in the main text, the rest in the supplement.
    dep_desc = (
        "in the OSA Severity Classifier "
        f"({targets_config['ahi_5']['desc'].replace(' @ ', ' ≥ ')})"
    )
    for plot_fn, dest_dir, fname in (
        (plot_shap_dependence_main, PATHS["fig_feature_interactions"], "figure5_shap_dependence.png"),
        (plot_shap_dependence_supplement, PATHS["fig_feature_interactions"], "figureS10_shap_dependence_extra.png"),
    ):
        plot_fn(
            full_shap["ahi_5"]["shap_values"],
            full_shap["ahi_5"]["X_test"],
            save_path=dest_dir / fname,
            target_desc=dep_desc,
            random_state=random_state,
        )

    # Interaction ranking that justifies the main/supplement split (supports Figure 5)
    model_ahi5, X_ahi5, _ = get_model_and_test(feature_sets, "all_features", "ahi_5")
    rank_interactions(model_ahi5, X_ahi5, display_names, random_state).to_csv(
        PATHS["tbl_feature_importance"] / "shap_interaction_ranking_ahi_5.csv",
        index=False,
    )
    print(f"SHAP dependence: {PATHS['fig_feature_interactions']}")

    # Stratum analyses (Age then BMI) for the three AHI binary targets.
    # ahi_5 panels are NOT used standalone — Figure 4 combines them into one
    # composite (age + BMI side by side); that composite is not produced by
    # this loop, so ahi_5 outputs below are draft-only. ahi_15/ahi_30 map
    # directly onto Supplementary Figures S11-S14.
    strata_destinations = {
        ("Age", "ahi_5"):  (PATHS["fig_draft"], "shap_beeswarm_age_ahi_5.png"),
        ("BMI", "ahi_5"):  (PATHS["fig_draft"], "shap_beeswarm_bmi_ahi_5.png"),
        ("Age", "ahi_15"): (PATHS["fig_feature_importance_stratified"], "figureS11_shap_age_ahi15.png"),
        ("BMI", "ahi_15"): (PATHS["fig_feature_importance_stratified"], "figureS13_shap_bmi_ahi15.png"),
        ("Age", "ahi_30"): (PATHS["fig_feature_importance_stratified"], "figureS12_shap_age_ahi30.png"),
        ("BMI", "ahi_30"): (PATHS["fig_feature_importance_stratified"], "figureS14_shap_bmi_ahi30.png"),
    }
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
            dest_dir, fname = strata_destinations[(stratum_label, target)]
            plot_shap_strata_beeswarm(
                groups_data,
                suptitle=f"SHAP Beeswarm by {stratum_label} Group — {desc}",
                save_path=dest_dir / fname,
                max_display=SHAP_MAX_DISPLAY,
            )
    print(f"Stratum figures saved → {PATHS['fig_feature_importance_stratified']} (+ draft: {PATHS['fig_draft']})")
    print(
        "NOTE: Figure 4 (figure4_shap_strata_age_bmi.png) is a hand-assembled "
        "composite of the Age and BMI AHI≥5 panels and is not regenerated by "
        "this script; see results/figures/feature_importance_stratified/figure4_shap_strata_age_bmi.png."
    )
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
