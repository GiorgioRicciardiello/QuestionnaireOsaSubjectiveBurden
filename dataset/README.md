# Dataset

This directory holds the input data referenced by `config/config.py`.

| File | Description | Source |
|------|-------------|--------|
| `sleep_research_cleaned.db` | SQLite database, 72,221 rows × 270 columns. Single table `sleep_data`. | Provided alongside the manuscript. |
| `encoding.json` | Mapping `raw_column_name → publication display name`. | Curated from the data dictionary. |

## Inclusion criterion

The pipeline filters with `dem_age >= 18` at load time.  Effective N
after filtering: ≈69,000 adult subjects.

## Schema highlights

| Group | Prefix | Examples |
|-------|--------|----------|
| Demographics | `dem_` | `dem_age`, `dem_gender`, `dem_bmi`, `dem_race` |
| Medical history | `mh_` | `mh_high_blood_pressure`, `mh_diabetes` |
| Sleep symptoms | `sa_` | `sa_sleep_snore`, `sa_sleep_stop_breath` |
| Epworth | `ep_` | `ep_score`, `ep_reading` |
| Pre-sleep / post-sleep | `presleep_`, `postsleep_` | `presleep_feel_alert_now`, `postsleep_feeling_refreshing` |
| OSA targets | `osa_` | `osa_binary_numeric`, `osa_four_numeric_rdi` |
| Engineered | `eng_is_*_missing` | Per-feature missingness indicators |

## Database not present?

The repository's `.gitignore` excludes the `.db` file because of size.
Place a copy of `sleep_research_cleaned.db` here before running any
entry point, or update `PATHS["database"]` in `config/config.py`.
