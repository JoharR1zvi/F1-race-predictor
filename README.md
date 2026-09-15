# F1 Race Outcome Predictor

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
notebooks/                       ← run in order, 01 → 02 → 03 → 04
├── 01_data_collection.ipynb          data pipeline from 3 APIs → master dataset
├── 02_Exploratory_Data_Analysis.ipynb exploratory data analysis → master_clean.csv
├── 03_feature_engineering.ipynb       rolling driver/constructor form features,
│                                        time-based train/test split → train.csv, test.csv
└── 04_model.ipynb                     baseline + Random Forest + XGBoost, trained on 3
                                        targets (points finish, podium, finish position),
                                        saves the winning model for each to the dashboard

app/
└── dashboard.py                  Streamlit dashboard -- EDA insights + live predictions
                                     across all 3 targets

models/                           saved by 04_model.ipynb, read by the dashboard
├── final_model.joblib                 winning pipeline for points finish (top 10)
├── model_metadata.json                its feature list, target, country codes, scores
├── model_comparison.csv               all 4 models' accuracy/F1/ROC-AUC for this target
├── podium_model.joblib                winning pipeline for podium (top 3)
├── podium_metadata.json               same, for the podium model
├── podium_model_comparison.csv        same, for the podium model
├── finish_position_model.joblib       winning pipeline for finish position (regression)
├── finish_position_metadata.json      same, for the finish-position model
└── finish_position_model_comparison.csv  same, for the finish-position model

data/
├── raw/                          source pulls straight from each API
├── processed/                    merged/cleaned datasets (master_dataset.csv, master_clean.csv,
│                                    train.csv, test.csv, feature_config.json)
├── interim/                      superseded/partial files kept for reference
└── figures/                      saved EDA + model evaluation plots

f1_cache/                         FastF1 local cache (do not delete, do not commit)
```

## Tech Stack
Python · Pandas · FastF1 · Scikit-learn · XGBoost · Matplotlib · Streamlit

## How to Run
```bash
pip install -r requirements.txt
```

## Run the Dashboard
The trained models and figures are already included in this repo, so the
dashboard runs right away, no need to re-run any notebook first:
```bash
streamlit run app/dashboard.py
```
Opens at `http://localhost:8501` -- an Insights tab (key EDA findings), a
Predict tab, and a Model Performance tab (how the winning model was chosen
for each target).

The Predict tab has two modes: **Simple**, pick a real driver and track
from the current grid and their actual recent form fills in automatically,
or **Advanced**, set all 22 model inputs by hand. Either way you get a
prediction for all 3 targets at once: points finish, podium, and finish
position.

## Reproducing the Data Pipeline
Raw and processed data files aren't included in this repo (too large,
regenerable). To rebuild everything from scratch, run the notebooks in
`notebooks/` in order (01 → 02 → 03 → 04). `01_data_collection.ipynb`
pulls from all 3 APIs and takes ~2 hours on a first run because of rate
limits (instant after caching).
