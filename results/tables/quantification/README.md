# Post-hoc quantification (no model re-run)

Inputs: `results/XGBoost/full_shap_table/*.csv`, reported point estimates in Supplementary Tables S4 and S6,
and label counts in `dataset/sleep_research_cleaned.db` (age >= 18, N = 67,397).

| File | Content |
|---|---|
| q1_shap_share.csv | Share of summed mean abs SHAP carried by the top 3 / top 1 features |
| q2_rank_stability.csv | Spearman rho and top-10 overlap of feature rankings between endpoints |
| q3_auroc_ci.csv | Hanley-McNeil 95% CIs for XGBoost AUROC |
| q3_auroc_diff_conservative.csv | AUROC difference CIs, treating the two AUROCs as independent (conservative) |
| q4_baselines.csv, q4_tabpfn_vs_majority.csv | Majority-class accuracy and TabPFN accuracy gain over it |
| q5_wilson_ci.csv | Wilson CIs for sensitivity, specificity, precision |
| supp_tables.json, supp_tables_spec.json | Content of Supplementary Tables S9-S11 |

Assumptions: nominal test set 6,740 (10% of 67,397; figures show 6,763-6,802), class prevalence of the full cohort,
Hanley-McNeil formula (assumes the reported AUROCs are from the same test set). A paired DeLong or bootstrap test
needs per-patient predictions, which are not in the project; replace these intervals if they become available.
