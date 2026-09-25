# F1 Race Outcome Predictor

A machine learning project that predicts Formula 1 race outcomes
using historical race data, telemetry, and weather conditions.

## What This Actually Does
Pick any driver on the current F1 grid, tell it where they're starting the
race, and it predicts three things: will they score points (finish in the
top 10), will they reach the podium (top 3), and roughly where will they
finish. It learned these patterns from 4 seasons of real race data
(2022-2025) - who started where, how fast they qualified, what the weather
was like, and how well each driver and team had been performing recently.

The interesting part isn't just "it makes a prediction." Three different
questions (points, podium, exact position) turned out to need three
different kinds of models to answer well, and the project explains why,
rather than picking one model and calling it done. It's also upfront about
where it struggles: predicting a driver's *exact* finishing position is
genuinely hard, a lap-1 crash isn't something any spreadsheet can see
coming, and the results say so plainly instead of hiding a weak number.

**Try it:** pick a real driver, hit Predict, see what happens — see "Run
the Dashboard" below.

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
└── 04_model.ipynb                     baseline through XGBoost, hyperparameter tuning with
                                        time-respecting cross-validation, SHAP explanations,
                                        trained on 3 targets (points finish, podium, finish
                                        position), saves the winning model for each

app/
└── dashboard.py                  Streamlit dashboard -- EDA insights + live predictions
                                     across all 3 targets

tests/                            pytest -- see "Running the Tests" below
├── test_dashboard.py                  clicks Predict for real, in both modes, using
│                                        Streamlit's AppTest framework
└── test_model_artifacts.py            sanity-checks the files models/ actually contains

models/                           saved by 04_model.ipynb, read by the dashboard
├── final_model.joblib                 winning pipeline for points finish (top 10)
├── model_metadata.json                its feature list, target, country codes, scores
├── model_comparison.csv               all 4 models' accuracy/F1/ROC-AUC for this target
├── podium_model.joblib                winning pipeline for podium (top 3)
├── podium_metadata.json               same, for the podium model
├── podium_model_comparison.csv        same, for the podium model
├── finish_position_model.joblib       winning pipeline for finish position (regression)
├── finish_position_metadata.json      same, for the finish-position model
├── finish_position_model_comparison.csv  same, for the finish-position model
├── driver_lookup.csv                  each driver's latest real race, for the dashboard's
│                                        Simple predict mode
└── circuit_profiles.csv               each circuit's historical average pace/weather,
                                        for the same

data/
├── raw/                          source pulls straight from each API
├── processed/                    merged/cleaned datasets (master_dataset.csv, master_clean.csv,
│                                    train.csv, test.csv, feature_config.json)
├── interim/                      superseded/partial files kept for reference
└── figures/                      saved EDA + model evaluation plots

f1_cache/                         FastF1 local cache (do not delete, do not commit)
```

## Tech Stack
Python · Pandas · FastF1 · Scikit-learn · XGBoost · SHAP · Matplotlib · Streamlit · pytest

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

## Running the Tests
```bash
pytest tests/
```
`test_dashboard.py` actually clicks Predict, in both Simple and Advanced
mode, using Streamlit's own `AppTest` framework -- no browser needed.
`test_model_artifacts.py` sanity-checks the files in `models/` (schema,
that every current driver and circuit is covered). Both exist because two
real bugs made it past manual testing earlier in this project: a
form missing several required inputs, and a numpy/Python float type
mismatch, and both only ever showed up once someone actually clicked the
button, not from the page just loading.

## Reproducing the Data Pipeline
Raw and processed data files aren't included in this repo (too large,
regenerable). To rebuild everything from scratch, run the notebooks in
`notebooks/` in order (01 → 02 → 03 → 04). `01_data_collection.ipynb`
pulls from all 3 APIs and takes ~2 hours on a first run because of rate
limits (instant after caching).

`01_data_collection.ipynb` needs `fastf1`, which isn't in the main
`requirements.txt` -- it requires `pandas<3.0`, which conflicts with the
`pandas==3.0.5` the dashboard and notebooks 02-04 are pinned to. Install
it in a **separate virtual environment** from the main one, on its own
(not combined with `requirements.txt` -- that would reintroduce the exact
conflict this avoids; `fastf1` pulls in its own compatible pandas/numpy
automatically):
```bash
python -m venv datacollection-env
datacollection-env\Scripts\activate  # or source .../bin/activate on macOS/Linux
pip install -r requirements-data-collection.txt jupyter
```
Everything else (the dashboard, notebooks 02-04, the tests) only ever
needs plain `requirements.txt`.
