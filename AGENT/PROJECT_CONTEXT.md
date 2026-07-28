# F1 Race Outcome Predictor — Master Project Context
## For Claude Code

This document gives you everything you need to understand, continue, and complete this project. Read it fully before touching any code.

---

## 1. Project Overview

**What this is:**
A machine learning project that predicts Formula 1 race outcomes using historical race data, lap telemetry, and weather conditions. Built as a portfolio project for a Masters in Data Science & Machine Learning student (Carl von Ossietzky Universität Oldenburg, Germany).

**The prediction targets:**
- `finish_position` — regression: predict exact finishing position (1–20)
- `on_podium` — binary classification: will driver finish top 3?
- `points_finish` — binary classification: will driver score points (top 10)?

**Primary target for initial modelling:** `points_finish` — it is perfectly balanced (50/50), cleanly defined, and practically meaningful.

**Why 2022 onwards only:**
2022 introduced completely new ground-effect car regulations — the most significant technical change in decades. Data before 2022 describes a fundamentally different car and racing environment. Using older data would hurt model accuracy.

**Tech stack:**
- Python 3.11 (Anaconda environment)
- pandas, numpy, matplotlib, seaborn, scipy
- fastf1 (F1 telemetry library)
- scikit-learn, XGBoost (modelling — not yet implemented)
- Jupyter Notebook

---

## 2. Project Structure

```
project_root/
├── data/
│   ├── raw/                          ← untouched source pulls, one file per API/endpoint
│   │   ├── jolpica_results.csv          ← race results from Jolpica API
│   │   ├── jolpica_qualifying.csv       ← qualifying times from Jolpica API
│   │   ├── jolpica_pitstops_agg.csv     ← aggregated pit stop data
│   │   ├── fastf1_laps.csv              ← lap-level telemetry from FastF1
│   │   ├── fastf1_weather.csv           ← weather data from FastF1
│   │   └── openf1_results.csv           ← driver nationality from OpenF1
│   ├── processed/                    ← merged/cleaned datasets
│   │   ├── master_dataset.csv           ← merged master (1838 rows × 50 cols)
│   │   └── master_clean.csv             ← cleaned version (output of 02_Exploratory_Data_Analysis.ipynb)
│   ├── interim/                      ← superseded/partial files kept for reference only
│   └── figures/                      ← saved EDA plots (eda_XX_description.png)
│
├── f1_cache/                        ← FastF1 local cache (DO NOT DELETE)
│
├── notebooks/
│   ├── 01_data_collection.ipynb                 ← COMPLETE ✅
│   ├── 02_Exploratory_Data_Analysis.ipynb        ← IN PROGRESS 🔄 (Section 6 onwards)
│   ├── 03_feature_engineering.ipynb              ← NOT STARTED ❌
│   └── 04_model.ipynb                            ← NOT STARTED ❌
│
└── AGENT/
    ├── PROJECT_CONTEXT.md           ← this file
    ├── DATA_DICTIONARY.md           ← every column documented
    ├── CONVENTIONS.md               ← coding style rules
    └── FINDINGS.md                  ← EDA findings and decisions so far
```

Note: both notebooks live in `notebooks/`, so all relative paths inside them are prefixed `../` (e.g. `../data/processed/master_dataset.csv`, `../f1_cache`). This is centralised in the `PATHS` dict in `02_Exploratory_Data_Analysis.ipynb` (Section 1) — update it there if the structure ever changes again.

---

## 3. Data Sources

### 3.1 Jolpica API
**URL:** `https://api.jolpi.ca/ergast/f1`
**What it is:** Community-maintained replacement for the deprecated Ergast F1 API. Same JSON format, different base URL.
**Coverage:** 1950–present (we use 2022–2025 only)
**Rate limits:** Polite usage, 0.5s sleep between calls, 30s sleep between seasons
**What we collect:**
- Race results (finish position, grid position, points, status, constructor)
- Qualifying times (Q1/Q2/Q3 times, qualifying position)
- Pit stops (stop number, lap, duration — aggregated to per-driver-per-race)

**Pagination:** API returns max 100 rows per call. Must paginate using `offset` parameter.

### 3.2 FastF1
**Library:** `fastf1` (pip install fastf1)
**What it is:** Python library that downloads official F1 timing data
**Coverage:** ~2018–present (we use 2022–2025)
**Rate limits:** F1 timing servers allow ~500 calls/hour. Sleep 5s between rounds, 120s between seasons.
**Cache:** All data cached locally in `f1_cache/` (project root; `../f1_cache` from within `notebooks/`) — subsequent loads are instant
**What we collect:**
- Lap-level data: lap times, sector times, tyre compound, tyre age, stint number
- Weather: air temp, track temp, humidity, wind speed, wind direction, pressure, rainfall

**Key FastF1 pattern:**
```python
sess = fastf1.get_session(year, round_number, 'R')
sess.load(telemetry=False, weather=True)
laps = sess.laps          # DataFrame of all laps
weather = sess.weather_data  # DataFrame of weather readings
```

**Known issue:** Weather data collected across multiple sessions had inconsistent column naming:
- Old collection: `mean_airtemp`, `max_airtemp` etc.
- New collection: `airtemp_mean`, `airtemp_max` etc.
- **Fix applied in 02_Exploratory_Data_Analysis.ipynb Section 3:** unified to `air_temp_mean`, `air_temp_max` etc.

### 3.3 OpenF1 API
**URL:** `https://api.openf1.org/v1`
**What it is:** Modern REST API for F1 data, free, no authentication for historical data
**Coverage:** 2023–present
**What we use it for:** ONLY `country_code` — the driver's nationality (3-letter ISO code e.g. 'NED', 'GBR')
**Why:** Jolpica doesn't provide driver nationality. Home race advantage is a real F1 phenomenon.
**Important:** 2022 rows will always have NaN for `country_code` — this is expected and correct.

---

## 4. Data Pipeline

### 4.1 Collection (01_data_collection.ipynb) — COMPLETE

**What was built:**
1. `get_race_results(year)` — paginates through Jolpica to get all race results for a season
2. `get_qualifying_results(year)` — gets Q1/Q2/Q3 times for all races
3. `get_pit_stops(year, round_num)` — gets pit stop data per race (has retry logic for connection resets)
4. `extract_lap_features(session)` — cleans FastF1 lap data, converts Timedelta to seconds
5. `aggregate_weather(session)` — summarises race weather into one row
6. `collect_fastf1_season(year)` — collects all FastF1 data for one season
7. `get_openf1_sessions(year)` — gets race session keys from OpenF1
8. `get_openf1_drivers(session_key)` — gets driver info including country_code
9. `merge_keeping_all(df_master, df_new, join_keys)` — smart merge that drops duplicate columns to prevent _x/_y suffixes

**Critical merge helper — use this everywhere:**
```python
def merge_keeping_all(df_master, df_new, join_keys):
    """
    Merge df_new into df_master on join_keys.
    Drops columns from df_new that already exist in df_master
    (except join keys) to prevent _x _y duplicate columns.
    """
    duplicate_cols = [
        c for c in df_new.columns
        if c in df_master.columns and c not in join_keys
    ]
    df_new_clean = df_new.drop(columns=duplicate_cols, errors='ignore')
    return df_master.merge(df_new_clean, on=join_keys, how='left')
```

**Master dataset build order:**
```
df_results (Jolpica)        ← SPINE: one row per driver per race
    + df_quali              ← join on year + round + driver_id
    + df_pitstops_agg       ← join on year + round + driver_id
    + df_fastf1_agg         ← join on year + round + driver_code
    + df_weather            ← join on year + round
    + openf1_useful         ← join on year + round + driver_code
```

**WARNING about OpenF1 join:** OpenF1 does NOT have a `round` column. It uses country names. We bridge via:
```python
country_name_fixes = {
    'United Kingdom':       'UK',
    'United Arab Emirates': 'UAE',
    'United States':        'USA',
}
# Then merge round_lookup (year + country → round) from df_results
```

**Master dataset shape:** 1838 rows × 50 columns

### 4.2 EDA (02_Exploratory_Data_Analysis.ipynb) — IN PROGRESS

**Completed sections:**
- Section 1: Setup & Configuration (COLUMN_CATALOGUE, STYLE, CONSTRUCTOR_COLOURS)
- Section 2: Data Overview (column audit table)
- Section 3: Data Cleaning (dtype fixes, derived columns)
- Section 4: Missing Value Analysis (bar chart + heatmap)
- Section 5: Target Variable Analysis (distributions, class balance)

**Currently stopped at:** Section 6 — Feature Distributions

**Sections remaining:**
- Section 6: Feature Distributions (histograms of all numeric columns)
- Section 7: Grid vs Finish Position
- Section 8: Pace Analysis
- Section 9: Constructor Effects
- Section 10: Weather & Strategy
- Section 11: Correlation Analysis
- Section 12: Summary & Decisions

---

## 5. Master Dataset — Column Reference

### Shape: 1838 rows × 50 columns
### Coverage: 2022–2025, 92 races, 31 drivers, 12 constructors

**Complete column list with roles:**

| Column | Source | Type | Role | Notes |
|--------|--------|------|------|-------|
| year | Jolpica | int | identifier | 2022-2025 |
| round | Jolpica | int | identifier | 1-24 depending on season |
| race_name | Jolpica | str | identifier | e.g. 'Bahrain Grand Prix' |
| circuit_id | Jolpica | str | identifier | e.g. 'bahrain' |
| circuit_name | Jolpica | str | identifier | e.g. 'Bahrain International Circuit' |
| country | Jolpica | str | identifier | e.g. 'Bahrain' |
| date | Jolpica | str→datetime | identifier | race date |
| driver_id | Jolpica | str | identifier | e.g. 'verstappen' |
| driver_code | Jolpica | str | identifier | e.g. 'VER' |
| driver_name | Jolpica | str | identifier | e.g. 'Max Verstappen' |
| constructor_id | Jolpica | str | identifier | e.g. 'red_bull' |
| constructor_name | Jolpica | str | identifier | e.g. 'Red Bull Racing' |
| grid_position | Jolpica | int | pre_race_feature | starting position |
| qual_position | Jolpica | float | pre_race_feature | qualifying position |
| q1_time | Jolpica | str | pre_race_feature | raw string e.g. '1:31.471' |
| q2_time | Jolpica | str | pre_race_feature | NaN if didn't make Q2 |
| q3_time | Jolpica | str | pre_race_feature | NaN if didn't make Q3 |
| total_pit_stops | Jolpica | float | post_race_leakage | 5.3% missing |
| avg_pit_duration | Jolpica | float | post_race_leakage | 6.7% missing |
| first_stop_lap | Jolpica | float | post_race_leakage | lap of first pit stop |
| avg_lap_time | FastF1 | float | post_race_leakage | 4.2% missing |
| best_lap_time | FastF1 | float | post_race_leakage | 4.2% missing |
| std_lap_time | FastF1 | float | post_race_leakage | lap time consistency |
| total_valid_laps | FastF1 | float | post_race_leakage | 4.2% missing |
| laps_on_soft | FastF1 | float | post_race_leakage | laps on soft tyre |
| laps_on_medium | FastF1 | float | post_race_leakage | laps on medium tyre |
| laps_on_hard | FastF1 | float | post_race_leakage | laps on hard tyre |
| avg_tyre_life | FastF1 | float | post_race_leakage | avg tyre age per lap |
| total_stints | FastF1 | float | post_race_leakage | number of tyre stints |
| had_rain | FastF1 | bool | post_race_leakage | was there rainfall |
| rain_laps | FastF1 | float | post_race_leakage | number of rainy readings |
| rain_fraction | FastF1 | float | post_race_leakage | proportion of race with rain |
| winddirection_mean | FastF1 | float | pre_race_feature | 29% missing, low value |
| country_code | OpenF1 | str | pre_race_feature | 51% missing (2022=NaN expected) |
| air_temp_mean | FastF1 | float | pre_race_feature | 1.1% missing |
| air_temp_min | FastF1 | float | pre_race_feature | 1.1% missing |
| air_temp_max | FastF1 | float | pre_race_feature | 1.1% missing |
| track_temp_mean | FastF1 | float | pre_race_feature | 1.1% missing |
| track_temp_min | FastF1 | float | pre_race_feature | 1.1% missing |
| track_temp_max | FastF1 | float | pre_race_feature | 1.1% missing |
| wind_speed_mean | FastF1 | float | pre_race_feature | 1.1% missing |
| wind_speed_max | FastF1 | float | pre_race_feature | 1.1% missing |
| finish_position | Jolpica | int | target | 1-20, regression target |
| points | Jolpica | float | target | 0-26, skewed (skew=1.39) |
| laps_completed | Jolpica | int | post_race_leakage | |
| status | Jolpica | str | post_race_leakage | 33 unique values |
| finished | Jolpica | bool | target | see important note below |
| on_podium | Jolpica | bool | target | 15% positive, imbalance 5.7:1 |
| points_finish | Jolpica | bool | target | 50% positive, perfectly balanced |

**Columns added during EDA cleaning (in master_clean.csv):**

| Column | Source | Role | Description |
|--------|--------|------|-------------|
| q1_time_s | derived | pre_race_feature | Q1 time converted to seconds |
| q2_time_s | derived | pre_race_feature | Q2 time converted to seconds |
| q3_time_s | derived | pre_race_feature | Q3 time converted to seconds |
| best_qual_time_s | derived | pre_race_feature | best of Q3→Q2→Q1 |
| track_temp_range | derived | pre_race_feature | track_temp_max - track_temp_min |
| positions_gained | derived | post_race_leakage | grid_position - finish_position |
| race_completion_pct | derived | post_race_leakage | laps_completed / max_laps_in_race |
| season_progress | derived | pre_race_feature | round / max_round_in_year (0→1) |
| top_10 | derived | target | finish_position <= 10 |

---

## 6. Critical Data Issues Found and Fixed

### Issue 1: Weather column naming inconsistency
**Problem:** Weather data was collected in two separate sessions using two different versions of `aggregate_weather()`. The old version named columns `mean_airtemp`, `max_airtemp` etc. The new version used `airtemp_mean`, `airtemp_max` etc. When stacked together, every row had NaN in one set of columns.

**Fix applied in 02_Exploratory_Data_Analysis.ipynb Section 3:**
```python
weather_fixes = {
    'air_temp_mean': ('airtemp_mean', 'mean_airtemp'),
    'air_temp_min':  ('airtemp_min',  'min_airtemp'),
    # etc.
}
for unified_name, (new_col, old_col) in weather_fixes.items():
    df_master[unified_name] = df_master[new_col].fillna(df_master[old_col])
```
**Result:** Weather coverage went from 28% to 98.9%

### Issue 2: `tracktempp_range` 71% missing
**Problem:** A typo in the original collection code (double 'p') created a broken column with 71% missing values.
**Fix:** Dropped the broken column. Recomputed as `track_temp_max - track_temp_min`.
**Result:** `track_temp_range` now has 98.9% coverage.

### Issue 3: `had_rain` stored as string
**Problem:** Boolean column was written to CSV as 'True'/'False' strings, read back as object dtype.
**Fix:**
```python
bool_map = {'True': True, 'False': False, True: True, False: False}
df_clean['had_rain'] = df_clean['had_rain'].map(bool_map)
```

### Issue 4: Qualifying times as strings
**Problem:** Q1/Q2/Q3 times stored as '1:31.471' strings — unusable for ML.
**Fix:** `time_string_to_seconds()` function converts to float seconds.
```python
def time_string_to_seconds(time_str):
    if pd.isna(time_str) or str(time_str).strip() == '':
        return np.nan
    try:
        parts = str(time_str).split(':')
        return float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[0])
    except:
        return np.nan
```

### Issue 5: `finished` column incorrectly defined
**Problem:** `finished = (status == 'Finished')` only. But 'Lapped', '+1 Lap', '+2 Laps' drivers also completed the race — they just weren't fast enough to stay on the lead lap. This caused DNF rate to show as 35% (wrong) instead of 14% (correct).
**Fix applied in 02_Exploratory_Data_Analysis.ipynb:**
```python
completed_statuses = ['Finished', 'Lapped', '+1 Lap', '+2 Laps']
df_clean['finished'] = df_clean['status'].isin(completed_statuses)
```
**Result:** Corrected finish rate 86.1%, DNF rate 13.9%

### Issue 6: OpenF1 many-to-many join explosion
**Problem:** Joining OpenF1 on `['year', 'driver_code']` caused row count to explode from 1838 to 31,451 because multiple OpenF1 rows per driver per year all matched multiple Jolpica rows.
**Fix:** Added `round` column to OpenF1 via country name matching, then joined on `['year', 'round', 'driver_code']`.
**Country name fixes needed:**
```python
country_name_fixes = {
    'United Kingdom':       'UK',
    'United Arab Emirates': 'UAE',
    'United States':        'USA',
}
```

### Issue 7: `top_10` and `points_finish` are identical
**Finding:** Both columns show exactly 50.1%/49.9% split — because scoring points in F1 means finishing top 10.
**Decision:** Drop `top_10` in feature engineering. Use `points_finish` as the primary binary target.

---

## 7. EDA Findings So Far

### 7.1 Target Variable Analysis
```
finish_position:  uniform distribution (skew=0.0), mean=10.5, range 1-20
points:           heavily right-skewed (skew=1.39), median=1.0, most drivers score 0
on_podium:        15% positive, imbalance 5.7:1 → needs class_weight="balanced"
points_finish:    50.1% positive, perfectly balanced → PRIMARY CLASSIFICATION TARGET
finished:         86.1% positive (after fix), real DNF rate ~14%
top_10:           identical to points_finish → DROP THIS COLUMN
```

### 7.2 Missing Values
```
q3_time:          51% missing — EXPECTED (only top 10 reach Q3)
country_code:     51% missing — EXPECTED (OpenF1 only has 2023+)
winddirection_mean: 29% missing — unreliable sensor, consider dropping
FastF1 columns:   4.2% missing — a few races not in FastF1 cache
pit stop columns: 5-7% missing — some races missing from Jolpica pit endpoint
```

### 7.3 Status Distribution (important for `finished` column fix)
```
Finished:         1200   → genuinely completed
Lapped:            297   → completed but behind leader (count as finished)
Retired:           153   → mechanical failure (genuine DNF)
+1 Lap:             80   → completed but one lap down (count as finished)
Collision damage:   14   → crash (genuine DNF)
Accident:           11   → crash (genuine DNF)
... (many other small categories)
```

---

## 8. Coding Conventions

**CRITICAL: The student is learning. Write code the simple, readable way first.**

### 8.1 Style Rules

**NO compressed code:**
```python
# ❌ Don't write this (too compressed for learning context)
result = [c for c in df.columns if df[c].dtype == bool]

# ✅ Write this instead
result = []
for c in df.columns:
    if df[c].dtype == bool:
        result.append(c)
```

**ALWAYS comment with WHY, not just WHAT:**
```python
# ❌ Bad comment
df_clean = df.copy()

# ✅ Good comment
# Work on a copy — never modify the original DataFrame
# If cleaning goes wrong, we can always restart from df without re-loading from disk
df_clean = df.copy()
```

**ALWAYS use defensive checks:**
```python
# Check columns exist before accessing them
if col in df_clean.columns:
    df_clean[col] = ...

# Use errors='ignore' when dropping
df_clean = df_clean.drop(columns=['col'], errors='ignore')
```

**NEVER hardcode column names outside the config:**
All column lists are derived from `COLUMN_CATALOGUE` at the top of 02_Exploratory_Data_Analysis.ipynb. Never scatter column names throughout the notebook.

### 8.2 Function Documentation Pattern
Every function uses this block:
```python
# ══════════════════════════════════════════════════════════
# WHAT:  one sentence describing what the function does
# WHY:   why we need it, what problem it solves
# INPUT: parameters and their types
# PLAN:
#   Step 1 → ...
#   Step 2 → ...
# OUTPUT: what it returns
# ══════════════════════════════════════════════════════════
def my_function(param):
    ...
```

### 8.3 Plot Style
All plots use the `STYLE` dictionary defined in Section 1 of 02_Exploratory_Data_Analysis.ipynb:
```python
STYLE = {
    'bg_dark':   '#0f0f0f',
    'bg_panel':  '#1a1a1a',
    'red':       '#e8002d',
    'white':     '#ffffff',
    'silver':    '#c0c0c0',
    'gold':      '#ffd700',
    'blue':      '#0093cc',
    'orange':    '#ff8000',
    'grid':      '#2a2a2a',
}
```

Constructor colours:
```python
CONSTRUCTOR_COLOURS = {
    'Red Bull Racing': '#3671c6',
    'Ferrari':         '#e8002d',
    'Mercedes':        '#00d2be',
    'McLaren':         '#ff8000',
    'Aston Martin':    '#358c75',
    'Alpine':          '#0093cc',
    'Williams':        '#64c4ff',
    'AlphaTauri':      '#5e8faa',
    'RB':              '#6692ff',
    'Alfa Romeo':      '#c92d4b',
    'Haas F1 Team':    '#b6babd',
}
```

All plots saved to `../data/figures/` (from within `notebooks/`) with naming convention `eda_XX_description.png`.

### 8.4 DataFrame Operations
**Always use pandas vectorisation, never row loops:**
```python
# ❌ Never do this
for i in range(len(df)):
    df.loc[i, 'new_col'] = df.loc[i, 'col_a'] - df.loc[i, 'col_b']

# ✅ Always do this
df['new_col'] = df['col_a'] - df['col_b']
```

**Use .apply() only when vectorisation is impossible (custom logic):**
```python
df['col_s'] = df['col'].apply(my_custom_function)
```

---

## 9. What Needs To Be Done Next

### 9.1 Complete 02_Exploratory_Data_Analysis.ipynb

Currently stopped at Section 6. Remaining sections:

**Section 6 — Feature Distributions**
Plot histograms for all numeric columns. Auto-detect columns (no hardcoding).
Flag columns with high skew (|skew| > 1.5) as candidates for log transform.

**Section 7 — Grid vs Finish Position**
The most important relationship in F1 analytics.
- Scatter plot grid vs finish (colour points by positions gained/lost)
- Average finish by grid slot with error bars
- Podium probability by grid slot
- Compute Pearson AND Spearman correlation (Spearman is more appropriate for rank data)
- Add statistical significance test (p-value)

**Section 8 — Pace Analysis**
IMPORTANT: Lap times MUST be normalised within each race before comparison.
Monaco laps are 78s, Monza laps are 80s — raw times are not comparable across circuits.
```python
# Normalise within each race
df['lap_time_z'] = df.groupby(['year', 'round'])['avg_lap_time'].transform(
    lambda x: (x - x.mean()) / x.std()
)
```
Then correlate normalised pace with finish position.

**Section 9 — Constructor Effects**
- Points per season per constructor (line chart, colour by constructor)
- Average finishing position by constructor (bar chart)
- DNF rate by constructor
- Use `CONSTRUCTOR_COLOURS` dict for colours
- Helper function to match constructor names to colours (names aren't always exact):
```python
def get_constructor_colour(name):
    for key, colour in CONSTRUCTOR_COLOURS.items():
        if key.lower() in name.lower() or name.lower() in key.lower():
            return colour
    return '#888888'
```

**Section 10 — Weather & Strategy**
- Welch t-test: positions gained in wet vs dry races (p-value determines significance)
- Position variance in wet vs dry races
- Pit stop count vs average finish (with error bars showing standard error NOT std dev)
- Track temperature bins vs finish position

**Section 11 — Correlation Analysis**
- Show correlation of ALL features with finish_position
- Colour bars by role (blue=pre_race, red=leakage, gold=target)
- Identify multicollinear feature pairs (|r| > 0.85)
- Show correlation heatmap of pre-race features only

**Section 12 — Summary & Decisions**
Auto-generate findings from computed values (no hardcoded numbers in the summary).
Must include feature engineering roadmap.

### 9.2 Build 03_feature_engineering.ipynb

Key things to engineer:

**Rolling features (most important):**
```python
# For each driver, compute rolling average of points over last N races
# Must be sorted by date first, then grouped by driver
# Use shift(1) to avoid using current race in the average (data leakage)
df_sorted = df.sort_values(['driver_id', 'date'])
df_sorted['rolling_points_5'] = (
    df_sorted.groupby('driver_id')['points']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
```

**Missing value strategy:**
```python
# q3_time_s: impute with worst qualifier's time in that race
# winddirection_mean: DROP (29% missing, weak predictor)
# country_code: leave as NaN, encode as 'UNKNOWN' category
# FastF1 features: impute with race-level mean
```

**Categorical encoding:**
```python
# constructor_name: target encoding (mean finish position per constructor)
# circuit_id: target encoding (some circuits suit certain teams)
# country_code: one-hot encode (limited unique values)
```

**Drop these columns before modelling:**
```python
# Post-race leakage (known only after race):
# avg_lap_time, best_lap_time, std_lap_time, total_pit_stops, etc.

# Identical columns:
# top_10 (same as points_finish)

# Low value:
# winddirection_mean (29% missing, weak correlation)
# q3_time_s (51% missing — use best_qual_time_s instead)

# Identifiers (not features):
# driver_id, circuit_name, race_name, etc.
```

**Do NOT use driver_id as a feature directly.** New drivers won't appear in training data. Use rolling averages that capture current form regardless of who the driver is.

### 9.3 Build 04_model.ipynb

**Cross-validation strategy: TIME-BASED ONLY — NOT RANDOM**
```python
# Train on 2022-2023, predict 2024
# Train on 2022-2024, predict 2025
# This mirrors real-world use: predict future races from past data
```

**Model order:**
1. Logistic Regression (baseline — simple, interpretable)
2. Random Forest (handles non-linearity, gives feature importance)
3. XGBoost (likely best performance)

**Evaluation metrics for on_podium (imbalanced):**
- Precision, Recall, F1 score
- ROC-AUC
- NOT just accuracy (misleading for imbalanced classes)

**For on_podium specifically:**
```python
# Must use class_weight='balanced' or equivalent
model = LogisticRegression(class_weight='balanced')
```

---

## 10. Important Things NOT To Do

1. **Never use random train/test split** — always time-based splits
2. **Never use post-race features as model inputs** — that's data leakage
3. **Never use driver_id directly as a feature** — new drivers problem
4. **Never compare raw lap times across circuits** — always normalise within race
5. **Never fill all NaNs with 0 or mean without understanding why they're missing**
6. **Never drop `top_10` before checking it's confirmed identical to `points_finish`**
7. **Never run all FastF1 seasons in one go** — hits rate limit, do one year at a time
8. **Never modify df directly** — always work on `df_clean = df.copy()`

---

## 11. Student Context

The student (Johar) is:
- A 2nd semester Masters student, new to practical data science projects
- Learning as he goes — explanations should be clear and educational
- Prefers simple readable code over compressed clever one-liners
- When showing compressed alternatives, label them clearly as "by the way, the compressed version would be..."
- Responds well to building up from simple examples before showing the full solution
- Is building this for his GitHub portfolio — code quality and documentation matter
- Previously worked as a Junior Technical Project Manager, has IT background

**Communication style that works:**
- Explain WHY before HOW
- Use concrete F1 examples (Verstappen, Bahrain GP etc.) not abstract examples
- When something is confusing, offer to explain it "like you're 12"
- Don't rush — understanding is more important than speed

---

## 12. Files Reference

**Read these CSVs at the start of each notebook (paths are relative to `notebooks/`):**
```python
df = pd.read_csv('../data/processed/master_clean.csv')    # use this for EDA and beyond
# OR
df = pd.read_csv('../data/processed/master_dataset.csv')  # raw merged version
```

**The `master_clean.csv` is the output of 02_Exploratory_Data_Analysis.ipynb and includes:**
- All cleaning fixes applied
- New derived columns added
- `finished` column corrected to include lapped drivers
- Weather columns unified

**Save all plots to:** `../data/figures/eda_XX_description.png`

**Save cleaned datasets to:** `./data/master_clean.csv` (EDA output), `./data/master_engineered.csv` (feature engineering output)
