"""SHAP analysis: computation, plots, importance tables, demographic strata."""

from .compute import (
    compute_shap_values,
    get_model_and_test,
    rename_features,
    exclude_feature_from_shap,
)
from .importance import calculate_feature_importance_table, export_importance_csv
from .plots import plot_shap_beeswarm_binary, plot_shap_beeswarm_multiclass, plot_shap_dependence_3x3
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
    "plot_shap_dependence_3x3",
    "shap_by_strata",
    "plot_shap_strata_beeswarm",
]
