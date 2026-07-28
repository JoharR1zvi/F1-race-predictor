# F1 Race Predictor — Data Dictionary
## Every column in master_dataset.csv and master_clean.csv

---

## Column Roles Explained

| Role | Meaning | Use in ML? |
|------|---------|------------|
| `identifier` | Labels that identify a row — who, where, when | NO — never feed into model |
| `pre_race_feature` | Known BEFORE the race starts — safe to use | YES |
| `post_race_leakage` | Only known AFTER the race — using it would be cheating | NO — data leakage |
| `target` | What we are trying to predict | YES — as labels |

---

## Identifier Columns (never use as model features)

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| `year` | int | 2023 | Season year |
| `round` | int | 1 | Race number within season |
| `race_name` | str | 'Bahrain Grand Prix' | Full official race name |
| `circuit_id` | str | 'bahrain' | Jolpica internal circuit ID |
| `circuit_name` | str | 'Bahrain International Circuit' | Full circuit name |
| `country` | str | 'Bahrain' | Country where race held |
| `date` | datetime | 2022-03-20 | Race date |
| `driver_id` | str | 'verstappen' | Jolpica driver ID (lowercase) |
| `driver_code` | str | 'VER' | 3-letter driver code |
| `driver_name` | str | 'Max Verstappen' | Full driver name |
| `constructor_id` | str | 'red_bull' | Jolpica constructor ID |
| `constructor_name` | str | 'Red Bull Racing' | Full constructor name |

---

## Pre-Race Features (safe to use in model)

These are all known before lights out on race day.

| Column | Type | Missing% | Description | Notes |
|--------|------|----------|-------------|-------|
| `grid_position` | int | 0% | Starting grid position | 0 = pit lane start |
| `qual_position` | float | 0.1% | Qualifying classification position | |
| `q1_time` | str | 1.1% | Q1 lap time as string '1:31.471' | Cleaned to `q1_time_s` in EDA |
| `q2_time` | str | 25.9% | Q2 lap time | NaN if didn't reach Q2 — EXPECTED |
| `q3_time` | str | 51.2% | Q3 lap time | NaN if didn't reach Q3 — EXPECTED |
| `q1_time_s` | float | 1.1% | Q1 time in seconds | Derived column, added in EDA |
| `q2_time_s` | float | 25.9% | Q2 time in seconds | Derived column, added in EDA |
| `q3_time_s` | float | 51.2% | Q3 time in seconds | Derived column, added in EDA |
| `best_qual_time_s` | float | 1.1% | Best of Q3→Q2→Q1 in seconds | Derived, use this not individual Q times |
| `air_temp_mean` | float | 1.1% | Mean air temperature during race (°C) | |
| `air_temp_min` | float | 1.1% | Min air temperature during race (°C) | |
| `air_temp_max` | float | 1.1% | Max air temperature during race (°C) | |
| `track_temp_mean` | float | 1.1% | Mean track surface temperature (°C) | Higher = tyres degrade faster |
| `track_temp_min` | float | 1.1% | Min track temperature (°C) | |
| `track_temp_max` | float | 1.1% | Max track temperature (°C) | |
| `track_temp_range` | float | 1.1% | track_temp_max - track_temp_min | Derived, replaces broken tracktempp_range |
| `wind_speed_mean` | float | 1.1% | Mean wind speed (m/s) | |
| `wind_speed_max` | float | 1.1% | Max wind speed (m/s) | |
| `winddirection_mean` | float | 29.4% | Mean wind direction (degrees) | HIGH MISSING — consider dropping |
| `country_code` | str | 51.1% | Driver nationality (ISO 3166 alpha-3) | 2022 always NaN — EXPECTED |
| `season_progress` | float | 0% | round / max_round_in_year (0→1) | Derived, added in EDA |

---

## Post-Race Leakage Columns (DO NOT use as model features)

These columns are only known after the race finishes. Using them to predict the race would be cheating — the model would be "looking into the future".

| Column | Type | Missing% | Description | Why it's leakage |
|--------|------|----------|-------------|-----------------|
| `laps_completed` | int | 0% | Number of laps completed | Only known after race |
| `status` | str | 0% | Race finish status | Only known after race |
| `avg_lap_time` | float | 4.2% | Mean lap time across race (seconds) | Only known after race |
| `best_lap_time` | float | 4.2% | Fastest lap of race (seconds) | Only known after race |
| `std_lap_time` | float | 4.4% | Std dev of lap times — consistency metric | Only known after race |
| `total_valid_laps` | float | 4.2% | Valid timed laps count | Only known after race |
| `laps_on_soft` | float | 4.2% | Laps completed on soft compound | Only known after race |
| `laps_on_medium` | float | 4.2% | Laps completed on medium compound | Only known after race |
| `laps_on_hard` | float | 4.2% | Laps completed on hard compound | Only known after race |
| `avg_tyre_life` | float | 4.2% | Average tyre age across all laps | Only known after race |
| `total_stints` | float | 4.2% | Number of tyre stints | Only known after race |
| `total_pit_stops` | float | 5.3% | Total number of pit stops | Only known after race |
| `avg_pit_duration` | float | 6.7% | Average pit stop duration (seconds) | Only known after race |
| `first_stop_lap` | float | 5.3% | Lap number of first pit stop | Only known after race |
| `had_rain` | bool | 1.1% | Was there rainfall during race? | Only known after race |
| `rain_laps` | float | 1.1% | Number of rainy weather readings | Only known after race |
| `rain_fraction` | float | 1.1% | Proportion of race with rain (0→1) | Only known after race |
| `positions_gained` | float | ~1% | grid_position - finish_position | Derived from finish, which is post-race |
| `race_completion_pct` | float | ~1% | laps_completed / max_race_laps | Derived from laps_completed |

**Important note on leakage:** Even though these columns are leakage for predicting race outcomes, they are valuable for:
1. EDA — understanding what actually determines race outcomes
2. Feature engineering — creating pre-race proxies from historical averages of these values

---

## Target Columns (what we predict)

| Column | Type | Positive% | Imbalance | Use For | Notes |
|--------|------|-----------|-----------|---------|-------|
| `finish_position` | int | N/A (regression) | N/A | Regression | Range 1-20, uniform distribution, skew=0.0 |
| `points` | float | N/A (regression) | N/A | Regression | Range 0-26, heavily skewed (skew=1.39), median=1.0 |
| `on_podium` | bool | 15.0% | 5.7:1 | Binary classification | NEEDS class_weight="balanced" |
| `points_finish` | bool | 50.1% | 1.0:1 | Binary classification | PRIMARY TARGET — perfectly balanced |
| `finished` | bool | 86.1% | 0.1:1 | Binary classification | After fix: includes Lapped/+1 Lap/+2 Laps |
| `top_10` | bool | 50.1% | 1.0:1 | — | IDENTICAL to points_finish — DROP before modelling |

---

## Status Column Values

The `status` column has 33 unique values. Here's how they map to `finished`:

**Count as Finished (finished=True):**
```
Finished      1200
Lapped         297
+1 Lap          80
+2 Laps          6
```

**Count as DNF (finished=False):**
```
Retired        153   (mechanical failure)
Collision damage 14
Accident        11
Collision       10
Disqualified    10
Did not start    9
Engine           7
Power Unit       6
Hydraulics       4
Gearbox          4
Undertray        3
Withdrew         3
Water pressure   2
Fuel pressure    2
Spun off         2
Water leak       2
... (other small categories)
```

**Real DNF rate: 13.9%** (not 35% as initially appeared)

---

## Source Files

| File | Shape | Description |
|------|-------|-------------|
| `jolpica_results.csv` | (1838, 20) | Raw race results from Jolpica |
| `jolpica_qualifying.csv` | (1838, 7) | Raw qualifying data |
| `jolpica_pitstops_agg.csv` | (1740, 6) | Aggregated pit stops (one row per driver per race) |
| `fastf1_laps.csv` | (100060, 46) | Raw lap-level data from FastF1 |
| `fastf1_weather.csv` | (91, 41) | Weather data (one row per race) |
| `openf1_results.csv` | (1463, 17) | OpenF1 data including country_code |
| `master_dataset.csv` | (1838, 50) | Merged master dataset |
| `master_clean.csv` | (1838, 55+) | Cleaned version with derived columns |
