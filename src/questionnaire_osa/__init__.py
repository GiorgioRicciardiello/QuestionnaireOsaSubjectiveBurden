"""questionnaire_osa
====================

Reproducible analysis pipeline for the manuscript

  *"Demographic versus questionnaire-derived predictors of OSA severity:
   a benchmark of XGBoost and TabPFN on 69,000 in-laboratory PSGs."*

Top-level sub-packages
----------------------
data            data loading & feature-set definitions
stratification  multilabel stratified split / k-fold
evaluation      classification metrics (binary + multiclass)
models          XGBoost (Optuna) and TabPFN trainers
shap_analysis   global, demographic-stratum, and dependence SHAP
plots           publication-quality matplotlib figures
reporting       text / CSV result summaries
utils           reproducibility helpers

Entry points (all use ``__main__``; no CLI args)
------------------------------------------------
* ``python -m questionnaire_osa.train_xgboost``
* ``python -m questionnaire_osa.train_tabpfn``
* ``python -m questionnaire_osa.analyze_xgboost``
* ``python -m questionnaire_osa.analyze_tabpfn``
"""

__version__ = "1.0.0"
