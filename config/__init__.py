"""Project configuration package.

All paths and runtime constants are imported from :mod:`config.config`
so that scripts never hard-code a filesystem location.
"""

from .config import (
    PATHS,
    SETTINGS,
    TARGETS,
    BINARY_TARGETS,
    MULTICLASS_TARGETS,
    ALL_TARGETS,
    FEATURE_PREFIXES,
    STRATIFICATION_CONFIG,
    FEATURE_SET_NAMES,
    AGE_GROUPS,
    BMI_GROUPS,
    CLASS_NAMES,
    FS_LABELS,
    FS_COLORS,
    FS_ORDER,
    SHAP_SAMPLE_SIZE,
    SHAP_MAX_DISPLAY,
)

__all__ = [
    "PATHS",
    "SETTINGS",
    "TARGETS",
    "BINARY_TARGETS",
    "MULTICLASS_TARGETS",
    "ALL_TARGETS",
    "FEATURE_PREFIXES",
    "STRATIFICATION_CONFIG",
    "FEATURE_SET_NAMES",
    "AGE_GROUPS",
    "BMI_GROUPS",
    "CLASS_NAMES",
    "FS_LABELS",
    "FS_COLORS",
    "FS_ORDER",
    "SHAP_SAMPLE_SIZE",
    "SHAP_MAX_DISPLAY",
]
