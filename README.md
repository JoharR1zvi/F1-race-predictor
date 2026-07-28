# 🏎️ F1 Race Outcome Predictor

A machine learning project that predicts Formula 1 race outcomes
using historical race data, telemetry, and weather conditions.

## Project Overview
Built as part of my MSc Data Science & Machine Learning at
Carl von Ossietzky Universität Oldenburg.

## Data Sources
- **Jolpica API** — Race results, qualifying times, pit stops (2022–2025)
- **FastF1** — Lap telemetry, tyre strategy, weather (2022–2025)
- **OpenF1 API** — Real-time race data (2023–2025)

## Project Structure
```
notebooks/                       ← run in order, 01 → 04
├── 01_data_collection.ipynb          data pipeline from 3 APIs → master dataset
├── 02_Exploratory_Data_Analysis.ipynb exploratory data analysis
├── 03_feature_engineering.ipynb       feature creation for ML (not started)
└── 04_model.ipynb                     model training and evaluation (not started)

data/
├── raw/                          source pulls straight from each API
├── processed/                    merged/cleaned datasets (master_dataset.csv, master_clean.csv)
├── interim/                      superseded/partial files kept for reference
└── figures/                      saved EDA plots

f1_cache/                         FastF1 local cache (do not delete, do not commit)

AGENT/                            project context docs for AI-assisted development
├── PROJECT_CONTEXT.md                 goals, data sources, scope decisions
├── DATA_DICTIONARY.md                 column-by-column reference
├── CONVENTIONS.md                     coding style rules
└── FINDINGS.md                        running EDA findings & decisions log
```

## Tech Stack
Python · Pandas · FastF1 · Scikit-learn · XGBoost · Matplotlib

## How to Run
```bash
pip install -r requirements.txt
```
Then run the notebooks in `notebooks/` in order (01 → 02 → 03 → 04).

Note: Data files are not included in this repo due to size.
Run `01_data_collection.ipynb` to generate them (takes ~2 hours
on first run due to API rate limits, instant after caching).
