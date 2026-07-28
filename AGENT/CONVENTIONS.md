# F1 Race Predictor — Coding Conventions
## Rules that must be followed throughout this project

---

## 1. Most Important Rule: Write Simple Code

The student is actively learning. ALWAYS write the simple, readable version first.

```python
# ❌ NEVER write compressed code without explanation
result = [c for c in df.columns if df[c].dtype == bool and c not in IDENTIFIERS]

# ✅ ALWAYS write the simple version
result = []
for c in df.columns:
    if df[c].dtype == bool:
        if c not in IDENTIFIERS:
            result.append(c)
```

After showing the simple version, you MAY mention:
> "By the way, experienced developers would compress this into one line using a list comprehension: `result = [c for c in df.columns if df[c].dtype == bool and c not in IDENTIFIERS]`"

But the simple version is always primary.

---

## 2. Comment Style

Every non-trivial line of code needs a comment explaining WHY, not just WHAT.

```python
# ❌ Useless comment — describes WHAT the code does (we can see that)
df_clean = df.copy()  # copy the dataframe

# ✅ Good comment — explains WHY we do this
# Work on a copy — never modify the original DataFrame.
# If cleaning goes wrong we can restart from df without re-loading the CSV.
df_clean = df.copy()
```

---

## 3. Function Documentation

Every function must have this block:
```python
# ══════════════════════════════════════════════════════════
# WHAT:  one sentence — what does this function do
# WHY:   why do we need it, what problem does it solve
# INPUT: parameter names and what they represent
# PLAN:
#   Step 1 → ...
#   Step 2 → ...
#   Step 3 → ...
# OUTPUT: what the function returns and its type
# ══════════════════════════════════════════════════════════
def my_function(year: int, round_num: int) -> pd.DataFrame:
    ...
```

---

## 4. No Hardcoded Values

Everything is defined once at the top of each notebook in a config section.

```python
# ❌ Never scatter column names throughout the notebook
df['avg_lap_time'].corr(df['finish_position'])  # hardcoded

# ✅ Define lists at the top, reference them everywhere
# (In config section at top of notebook)
NUMERIC_FEATURES = ['avg_lap_time', 'grid_position', ...]

# (Later in notebook)
df[NUMERIC_FEATURES[0]].corr(df['finish_position'])
```

The `COLUMN_CATALOGUE` in `02_Exploratory_Data_Analysis.ipynb` is the single source of truth for all column metadata.

---

## 5. DataFrame Rules

**Always copy before modifying:**
```python
df_clean = df.copy()  # always work on a copy
```

**Use vectorised operations, not row loops:**
```python
# ❌ Never loop through rows
for i in range(len(df)):
    df.loc[i, 'result'] = df.loc[i, 'a'] - df.loc[i, 'b']

# ✅ Use vectorised pandas operations
df['result'] = df['a'] - df['b']
```

**Use .apply() only when custom logic is required:**
```python
# Only use .apply() when vectorisation is impossible
df['col_s'] = df['col'].apply(time_string_to_seconds)
```

**Defensive checks before column access:**
```python
if col in df_clean.columns:
    df_clean[col] = ...

df_clean = df_clean.drop(columns=['col'], errors='ignore')
```

---

## 6. Merge Convention

Always use `merge_keeping_all()` — never raw `.merge()` — to prevent _x/_y duplicate columns:

```python
def merge_keeping_all(df_master, df_new, join_keys):
    """Merges df_new into df_master, dropping columns that already exist."""
    duplicate_cols = [
        c for c in df_new.columns
        if c in df_master.columns and c not in join_keys
    ]
    df_new_clean = df_new.drop(columns=duplicate_cols, errors='ignore')
    return df_master.merge(df_new_clean, on=join_keys, how='left')
```

---

## 7. Plot Style

All plots use the STYLE dict:
```python
STYLE = {
    'bg_dark':  '#0f0f0f',
    'bg_panel': '#1a1a1a',
    'red':      '#e8002d',
    'white':    '#ffffff',
    'silver':   '#c0c0c0',
    'gold':     '#ffd700',
    'blue':     '#0093cc',
    'orange':   '#ff8000',
    'grid':     '#2a2a2a',
}
```

All plots saved to `../data/figures/eda_XX_description.png` (paths are relative to `notebooks/`, where the notebooks live).

Figure backgrounds: `facecolor=STYLE['bg_dark']`
Panel backgrounds: set in `plt.rcParams`

---

## 8. API Rate Limiting

**Jolpica:** 0.5s between calls, 0.3s between pagination calls
**FastF1:** 5s between rounds, 120s between seasons
**OpenF1:** 0.5s between calls, 5s between seasons

Always include `time.sleep()` calls. Always include retry logic for connection errors:
```python
for attempt in range(3):
    try:
        response = requests.get(url, timeout=10)
        # process response
        break
    except Exception as e:
        wait = (attempt + 1) * 10
        time.sleep(wait)
```

---

## 9. File Paths

```python
PATHS = {
    'input':      '../data/processed/master_dataset.csv',
    'clean':      '../data/processed/master_clean.csv',
    'engineered': '../data/processed/master_engineered.csv',
    'plots':      '../data/figures/',
}
```

Never hardcode file paths inline. Paths are relative to `notebooks/`, where all notebooks live.

---

## 10. Cross Validation Rule

**NEVER use random train/test split on this dataset.**
Always use time-based splits:

```python
# ✅ Correct — train on past, predict future
train = df[df['year'] < 2024]
test  = df[df['year'] == 2024]

# ❌ Wrong — mixes years, allows future information into training
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)  # DO NOT DO THIS
```

---

## 11. Feature Leakage Prevention

Before any model training, drop ALL post-race leakage columns:
```python
LEAKAGE_COLS = [c for c, m in COLUMN_CATALOGUE.items()
                if m['role'] == 'post_race_leakage']

X = df.drop(columns=LEAKAGE_COLS + TARGETS + IDENTIFIERS, errors='ignore')
```
## 12. Modular Code — No Redundancy

**Every piece of logic is written once and reused. Never copy-paste code.**

### Functions over repetition
If you find yourself writing the same logic twice, it becomes a function immediately.

```python
# ❌ Never repeat logic
df_2022 = df[df['year'] == 2022].groupby('constructor_name')['points'].sum()
df_2023 = df[df['year'] == 2023].groupby('constructor_name')['points'].sum()
df_2024 = df[df['year'] == 2024].groupby('constructor_name')['points'].sum()

# ✅ Write it once as a function
def get_constructor_points(df, year):
    return df[df['year'] == year].groupby('constructor_name')['points'].sum()

df_2022 = get_constructor_points(df, 2022)
df_2023 = get_constructor_points(df, 2023)
df_2024 = get_constructor_points(df, 2024)
```

### Config drives everything
If a value appears more than once in the code, it belongs in the config at the top.

```python
# ❌ Never scatter magic numbers or column names
if df['null_pct'] > 50:   # why 50? defined where?
    ...
df['air_temp_mean']       # hardcoded column name

# ✅ Define once at top, reference everywhere
MISSING_THRESHOLD = 50
WEATHER_COLS = ['air_temp_mean', 'track_temp_mean', ...]

if df['null_pct'] > MISSING_THRESHOLD:
    ...
df[WEATHER_COLS[0]]
```

### One cell, one responsibility
Each notebook cell does exactly one thing. Never combine data loading, cleaning, and plotting in the same cell.
Cell 1 → load data
Cell 2 → clean data
Cell 3 → compute statistics
Cell 4 → plot statistics

### Loops over copy-paste for similar charts
If you are creating similar plots for multiple variables, loop over them — do not make one cell per variable.

```python
# ❌ Never do this
ax1.hist(df['air_temp_mean'], ...)
ax2.hist(df['track_temp_mean'], ...)
ax3.hist(df['wind_speed_mean'], ...)

# ✅ Loop over the columns
weather_cols = ['air_temp_mean', 'track_temp_mean', 'wind_speed_mean']
for idx, col in enumerate(weather_cols):
    axes[idx].hist(df[col], ...)
```

### Helper functions live at the top of the notebook
Any reusable function (plotting helpers, data transformers, metric computers) is defined in the Setup section at the top — not inline where it's first used.

### DRY principle — Don't Repeat Yourself
If the same transformation, filter, or plot appears more than once, it is wrong. Refactor it into a function or loop immediately.