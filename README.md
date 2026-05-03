# QuestionnaireOsaSubjectiveBurden

Reproducible analysis pipeline for the manuscript

> **Demographic versus questionnaire-derived predictors of OSA severity:
> a benchmark of XGBoost and TabPFN on 69,000 in-laboratory PSGs.**

The cohort comprises 69,000 adults referred for in-laboratory polysomnography
(PSG).  Three feature configurations — *All Features* (169 vars),
*Questionnaire-Only* (~161 vars), and *Demographics-Only* (4 vars: age,
gender, BMI, race) — are compared against AHI/RDI binary thresholds
(5/15/30 events·h⁻¹) and against a 4-class severity target.

Two model families are evaluated:

| Model | Configuration | Output |
|-------|---------------|--------|
| **XGBoost** | Optuna search (100 trials), 90/10 outer split, 5-fold inner CV | `results/xgboost_cv_results.pkl` |
| **TabPFN v2.5** | Hyper-parameter-free, 5-fold CV (no held-out test) | `results/tabpfn_cv_results.pkl` |

SHAP analysis is reported globally and within predefined age (≤36, 36–46,
46–58, ≥58) and BMI (≤26, 26–30, 30–34, ≥34) strata.

---
Background: Validated questionnaires are widely used to screen for obstructive sleep apnea (OSA), yet the incremental value of patient-reported symptom data beyond routinely available demographic variables remains poorly characterised. We aimed to quantify the discriminative contribution of questionnaire-derived items relative to demographic variables alone for OSA severity classification.

Methods: We analysed a retrospective cohort of 69,000 adults referred for in-laboratory polysomnography (PSG). XGBoost classifiers were trained under three feature configurations: All-Features (169 variables), Questionnaire-Only (161 variables), and Demographics-Only (age, gender, BMI, and race; 4 variables). A transformer-based tabular foundation model (TabPFN v2.5) was evaluated as a secondary comparator. Binary classification was assessed at apnea–hypopnea index (AHI) and respiratory disturbance index (RDI) thresholds of 5, 15, and 30 events/h. SHapley Additive exPlanations (SHAP) analysis was performed globally and within predefined age and BMI strata.

Results: The Demographics-Only model achieved AUROC values of 0.755, 0.729, and 0.750 at AHI thresholds of 5, 15, and 30 events/h, respectively, differing from the All-Features model by only 0.020–0.033 AUROC units. The Questionnaire-Only model performed worst at all thresholds (AUROC 0.706–0.708), with reductions of 0.054–0.076 relative to the All-Features model. SHAP identified age, BMI, and gender as the three dominant predictors at all thresholds across the full feature set. Stratified analysis revealed a systematic age–BMI interaction: age dominated in younger, leaner patients whereas BMI dominated in those with higher adiposity. Four-class severity classification yielded macro-averaged F1 scores ≤0.40 across all configurations, indicating that fine-grained stratification from non-PSG data remains beyond the resolution of any feature set evaluated.

Conclusions: Questionnaire-derived symptom data provides negligible incremental discriminative value for OSA severity classification when demographic information is available. Four readily accessible variables account for the substantial majority of achievable predictive performance, with direct implications for the design of scalable, low-burden screening pathways



---

## Results

**ROC Curves on Test Set (Threshold = 0.50)**

<div align="center">
  <img src="figures/figure 1.png" width="1000px" alt="ROC Curves comparing Demographics Only, Questionnaire Only, and All Features across AHI and RDI thresholds">
</div>

The figure shows model performance across six severity thresholds (AHI ≥ 5/15/30 and RDI ≥ 5/15/30). The **All Features** model (green) consistently outperforms **Demographics Only** (blue) and **Questionnaire Only** (orange) in AUC, sensitivity, and specificity across all thresholds.

---

## Stratification

The 90/10 outer split and the 5-fold inner CV are jointly stratified
on:

* `dem_age` (continuous, bins `[0, 30, 40, 50, 60, 70, 100]`)
* `dem_gender` (categorical)
* `dem_bmi` (continuous, bins `[0, 18.5, 25, 30, 35, 40, 100]`)
* `dem_race`, `sa_sleep_snore`, `sa_sleep_stop_breath`,
  `LAB_CAT`, `MD_Identifyer` (categorical)
* the target itself

This guarantees that every fold and the held-out test set match the
case-mix of the full cohort.

---
___

## Project layout

```
QuestionnaireOsaSubjectiveBurden/
├── config/                 # Path & runtime configuration (single source of truth)
│   └── config.py
├── dataset/                # SQLite database + encoding mappings
│   ├── sleep_research_cleaned.db
│   ├── encoding.json
│   └── README.md
├── results/                # Pickled training artefacts (auto-created)
├── figures/                # Output plots, organised by analysis type
├── tables/                 # CSV summaries
└── src/questionnaire_osa/  # Importable Python package
    ├── data/               # SQLite loader + feature-set builder
    ├── stratification/     # Multilabel stratified splitter / k-fold
    ├── evaluation/         # Classification metrics
    ├── models/             # XGBoost (+Optuna) and TabPFN trainers
    ├── shap_analysis/      # SHAP compute, plots, importance, strata
    ├── plots/              # ROC, confusion, distribution figures
    ├── reporting/          # Result-summary CSV / text
    ├── utils/              # Reproducibility helpers
    ├── train_xgboost.py    # entry point
    ├── train_tabpfn.py     # entry point
    ├── analyze_xgboost.py  # entry point
    └── analyze_tabpfn.py   # entry point
```

All filesystem paths live in **`config/config.py`**.  Scripts never embed
strings of their own; if you need to change a path, change it in one
place.

---

## Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate questionnaire_osa
pip install -e .
```

### Option B — pip + virtualenv

```bash
python -m venv .venv
. .venv/Scripts/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### TabPFN authentication

TabPFN v2.5 requires a Prior Labs token.  Set it before running:

```bash
# Windows (PowerShell)
$env:TABPFN_TOKEN = "your-token-here"

# Linux / macOS
export TABPFN_TOKEN=your-token-here
```

---

## Reproducing the manuscript

The pipeline is split into four entry points; each is invoked as a
module so that the package's `config/` is on the path.  None take CLI
arguments — every knob lives in `config/config.py`.

```bash
# 1. Train all 24 XGBoost models  (≈ 5–6 h, CPU)
python -m questionnaire_osa.train_xgboost

# 2. Cross-validate TabPFN v2.5 across all 8 targets  (≈ 30–60 min, GPU)
python -m questionnaire_osa.train_tabpfn

# 3. Regenerate every XGBoost figure & table from the pickle
python -m questionnaire_osa.analyze_xgboost

# 4. Regenerate TabPFN figures
python -m questionnaire_osa.analyze_tabpfn
```

Steps 1 and 2 produce `results/*.pkl`; steps 3 and 4 read those pickles
and emit `figures/**/*.png` plus `tables/*.csv`.  All artefacts are
deterministic given a fixed seed (default `42`).

---

## Targets

8 targets, defined in `config.config.TARGETS`:

| Key | Database column | Type | Group | Description |
|-----|-----------------|------|-------|-------------|
| `ahi_5` | `osa_binary_numeric` | binary | AHI | AHI ≥ 5 |
| `ahi_15` | `osa_binary_fifteenth_numeric` | binary | AHI | AHI ≥ 15 |
| `ahi_30` | `osa_binary_thirty_numeric` | binary | AHI | AHI ≥ 30 |
| `rdi_5` | `osa_binary_numeric_rdi` | binary | RDI | RDI ≥ 5 |
| `rdi_15` | `osa_binary_fifteenth_numeric_rdi` | binary | RDI | RDI ≥ 15 |
| `rdi_30` | `osa_binary_thirty_numeric_rdi` | binary | RDI | RDI ≥ 30 |
| `ahi_4class` | `osa_four_numeric` | multiclass | AHI | AHI 4-class severity |
| `rdi_4class` | `osa_four_numeric_rdi` | multiclass | RDI | RDI 4-class severity |

---


## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this code or its results, please cite the associated
manuscript.
