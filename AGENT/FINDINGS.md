# F1 Race Predictor — EDA Findings & Decisions Log
## Running record of everything discovered and decided

---

## Data Quality Issues Found & Fixed

### Fix 1: Weather Column Naming Inconsistency
**What happened:** FastF1 weather was collected in multiple sessions. The old `aggregate_weather()` function named columns `mean_airtemp`, `max_airtemp`. The new version used `airtemp_mean`, `airtemp_max`. When stacked together, most rows had NaN in one set of columns.

**Impact:** Weather coverage appeared as 28% but was actually 98.9%.

**Fix applied:** Unified all weather columns to consistent naming:
```
air_temp_mean, air_temp_min, air_temp_max
track_temp_mean, track_temp_min, track_temp_max
wind_speed_mean, wind_speed_max
```
Used `.fillna()` to merge old and new column values.

**Result:** Weather coverage corrected to 98.9%.

---

### Fix 2: tracktempp_range — 71% Missing (Typo)
**What happened:** Column name had a typo (double 'p' — `tracktempp_range`). The old and new weather functions didn't align, causing 71% NaN.

**Fix applied:** Dropped broken column. Recomputed cleanly:
```python
df_clean['track_temp_range'] = df_clean['track_temp_max'] - df_clean['track_temp_min']
```

**Result:** `track_temp_range` has 98.9% coverage.

---

### Fix 3: had_rain Stored as String
**What happened:** Boolean column written to CSV as 'True'/'False' strings, read back as object dtype.

**Fix applied:**
```python
bool_map = {'True': True, 'False': False, True: True, False: False}
df_clean['had_rain'] = df_clean['had_rain'].map(bool_map)
```

---

### Fix 4: Qualifying Times as Strings
**What happened:** Q1/Q2/Q3 times stored as '1:31.471' — unusable for ML.

**Fix applied:** `time_string_to_seconds()` converts '1:31.471' → 83.456 (total seconds).

New columns created: `q1_time_s`, `q2_time_s`, `q3_time_s`, `best_qual_time_s`
Old string columns dropped.

---

### Fix 5: finished Column Incorrectly Defined — CRITICAL
**What happened:** `finished` was defined as `status == 'Finished'` only. This caused:
- Lapped drivers (297 rows) to be counted as DNF
- +1 Lap drivers (80 rows) to be counted as DNF
- +2 Laps drivers (6 rows) to be counted as DNF

**Impact:** DNF rate appeared as 35% instead of the real 14%.

**Status breakdown discovered:**
```
Finished:    1200  → completed race
Lapped:       297  → finished but behind leader (SHOULD COUNT AS FINISHED)
Retired:      153  → genuine DNF
+1 Lap:        80  → one lap down (SHOULD COUNT AS FINISHED)
+2 Laps:        6  → two laps down (SHOULD COUNT AS FINISHED)
```

**Fix applied:**
```python
completed_statuses = ['Finished', 'Lapped', '+1 Lap', '+2 Laps']
df_clean['finished'] = df_clean['status'].isin(completed_statuses)
```

**Result:** Corrected finish rate 86.1%, DNF rate 13.9% — realistic for modern F1.

---

### Fix 6: OpenF1 Many-to-Many Join Explosion
**What happened:** Joining OpenF1 on `['year', 'driver_code']` caused row count to explode from 1838 to 31,451. Each driver had ~22 OpenF1 rows per year and ~22 Jolpica rows — 22×22 = 484 rows per driver per year.

**Root cause:** OpenF1 doesn't have a `round` column. We were joining on only year+driver which wasn't unique.

**Fix applied:** Added `round` to OpenF1 by matching country names:
```python
country_name_fixes = {
    'United Kingdom':       'UK',
    'United Arab Emirates': 'UAE',
    'United States':        'USA',
}
# Then joined round_lookup (year + country → round) from df_results
# Then joined OpenF1 on ['year', 'round', 'driver_code']
```

**Result:** Row count stayed at 1838.

---

## Key EDA Findings

### Finding 1: Finish Position is Uniformly Distributed
`finish_position` has skew=0.0, mean=10.5, range 1-20.
Every position appears roughly equally often (~92 times each across 92 races).
This is expected — there is exactly one driver per position per race.
**Decision:** No transformation needed for regression target.

---

### Finding 2: Points is Heavily Skewed
`points` has skew=1.39, median=1.0, mean=5.1.
Massive spike at 0 — 10 of 20 drivers score 0 every race.
**Decision:** If modelling points directly, consider log1p transformation. More likely: just use `points_finish` (binary) instead, which is perfectly balanced.

---

### Finding 3: top_10 and points_finish are Identical
Both show exactly 50.1%/49.9% split.
**Reason:** In modern F1, scoring points means finishing top 10. These are the same thing.
**Decision:** DROP `top_10` before modelling. Use `points_finish` as primary binary target.

---

### Finding 4: on_podium is Moderately Imbalanced
Only 15% positive rate, imbalance 5.7:1.
**Decision:** Must use `class_weight='balanced'` in any classifier. Cannot use accuracy as metric — use F1/ROC-AUC.

---

### Finding 5: Real DNF Rate is 14%, Not 35%
Discovered when investigating `finished` column (see Fix 5 above).
**Significance:** This is a real data quality issue that would have corrupted any model trained to predict race completion.

---

## Decisions Made

### Decision 1: Primary ML Target
**Chosen:** `points_finish` (binary classification)
**Reason:** Perfectly balanced (50/50), practically meaningful, cleanest problem to start with.
**Secondary targets:** `on_podium` (harder, imbalanced), `finish_position` (regression, later)

### Decision 2: Don't Use driver_id as Feature
**Reason:** New drivers in future seasons won't appear in training data. Model should learn from situation (rolling form, grid position, constructor) not identity.
**Alternative:** Rolling averages that capture current form regardless of driver identity.

### Decision 3: Time-Based Cross Validation Only
**Train:** 2022-2023 → **Predict:** 2024
**Train:** 2022-2024 → **Predict:** 2025
**Reason:** Mirrors real-world use. Random splits would allow future data to leak into training.

### Decision 4: Columns to Drop Before Modelling
```
top_10               — identical to points_finish
winddirection_mean   — 29% missing, weak predictor, unreliable sensor
q3_time_s            — 51% missing, use best_qual_time_s instead
All LEAKAGE_COLS     — post-race data, cannot use to predict race
All IDENTIFIERS      — driver_id, circuit_name etc. are labels not features
```

### Decision 5: OpenF1 Scope
OpenF1 had 17 columns. Only `country_code` was genuinely new — everything else duplicated Jolpica.
URLs, hex colours, first/last names are useless for ML.

---

## Feature Engineering Roadmap (for 03_feature_engineering.ipynb)

### Priority 1 — Rolling Features (most impactful)
```python
# Driver rolling average points (last 3, 5, 10 races)
# Must use shift(1) to avoid current race leaking into the feature
df_sorted = df.sort_values(['driver_id', 'date'])
df_sorted['rolling_points_5'] = (
    df_sorted.groupby('driver_id')['points']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Constructor rolling performance
# Driver rolling DNF rate
# Driver rolling positions gained/lost
```

### Priority 2 — Categorical Encoding
```python
# constructor_name → target encoding (avg finish position per constructor)
# circuit_id → target encoding (circuits suit certain teams/styles)
# country_code → one-hot encode (limited unique values, 17 nationalities)
```

### Priority 3 — Missing Value Strategy
```python
# q3_time_s (51% missing):
#   Impute with worst qualifier's time in that race
#   Rationale: drivers who don't make Q3 are slower

# winddirection_mean (29% missing):
#   DROP entirely — unreliable, weak predictor

# FastF1 features (4.2% missing):
#   Impute with race-level mean
#   These few missing races are random, not systematic

# country_code (51% missing for 2022):
#   Leave as NaN for 2022, encode 'UNKNOWN' category
#   Model will learn 2022 = no nationality data
```

### Priority 4 — Feature Transforms
```python
# Log transform for heavily skewed features if needed
# import numpy as np
# df['points_log'] = np.log1p(df['points'])
```

---

## Sections Completed in 02_Exploratory_Data_Analysis.ipynb

- [x] Section 1: Setup & Configuration
- [x] Section 2: Data Overview (column audit)
- [x] Section 3: Data Cleaning
- [x] Section 4: Missing Value Analysis
- [x] Section 5: Target Variable Analysis
- [x] Section 6: Feature Distributions
- [ ] Section 7: Grid vs Finish Position ← NEXT
- [ ] Section 8: Pace Analysis (normalise within race!)
- [ ] Section 9: Constructor Effects
- [ ] Section 10: Weather & Strategy
- [ ] Section 11: Correlation Analysis
- [ ] Section 12: Summary & Decisions

Note: `master_clean.csv` has not been saved yet — that only happens at the end of Section 12. Notebook 03 will fail with `FileNotFoundError` until EDA is finished.
