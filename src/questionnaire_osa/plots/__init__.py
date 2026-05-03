"""Publication-quality matplotlib figures."""

from .style import apply_publication_style, add_panel_labels
from .roc import plot_roc_curves
from .confusion import (
    plot_confusion_matrices_binary,
    plot_confusion_matrices_multiclass,
)
from .stratification import (
    plot_distribution_comparison,
    plot_target_distribution,
    plot_strata_panel,
)

__all__ = [
    "apply_publication_style",
    "add_panel_labels",
    "plot_roc_curves",
    "plot_confusion_matrices_binary",
    "plot_confusion_matrices_multiclass",
    "plot_distribution_comparison",
    "plot_target_distribution",
    "plot_strata_panel",
]
