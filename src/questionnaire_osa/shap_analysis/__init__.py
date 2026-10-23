"""SHAP analysis: computation, plots, importance tables, demographic strata."""

from .compute import (
    compute_shap_values,
    get_model_and_test,
    rename_features,
    exclude_feature_from_shap,
)
from .importance import calculate_feature_importance_table, export_importance_csv
from .composite import plot_shap_beeswarm_models, plot_shap_strata_combined
from .dependence import plot_shap_dependence_main, plot_shap_dependence_supplement
from .interactions import rank_interactions
from .plots import plot_shap_beeswarm_binary, plot_shap_beeswarm_multiclass
from .strata import shap_by_strata, plot_shap_strata_beeswarm

__all__ = [
    "compute_shap_values",
    "get_model_and_test",
    "rename_features",
    "exclude_feature_from_shap",
    "calculate_feature_importance_table",
    "export_importance_csv",
    "plot_shap_beeswarm_binary",
    "plot_shap_beeswarm_multiclass",
    "plot_shap_beeswarm_models",
    "plot_shap_strata_combined",
    "plot_shap_dependence_main",
    "plot_shap_dependence_supplement",
    "rank_interactions",
    "shap_by_strata",
    "plot_shap_strata_beeswarm",
]
