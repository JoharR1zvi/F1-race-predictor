"""
F1 Race Outcome Predictor -- Dashboard

Shows the same story the notebooks already told, in one place: what the
EDA found (Insights), what three separately trained models predict for a
real driver or a race you make up yourself (Predict), and how each of
those models was chosen (Model Performance). Three targets now, all
sharing the same 22 pre-race features: will this driver finish in the
points, will they reach the podium, and roughly where will they finish.

Nothing in this file computes anything the notebooks haven't already
computed and saved -- it only loads and displays. That's a deliberate
choice: if a number here disagrees with the notebooks, something is
wrong, and it should be easy to tell which side is stale.
"""

from pathlib import Path

import joblib
import json
import pandas as pd
import streamlit as st

# ── Paths ─────────────────────────────────────────────────────────────
# Path(__file__) always points at this file's own location, no matter
# where `streamlit run` gets launched from -- unlike a relative path,
# which depends on the current working directory.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
FIGURES_DIR = BASE_DIR / "data" / "figures"

# One entry per target `04_model.ipynb` trained a model for. Everything
# below (loading, the at-a-glance strip, the Predict tab's result cards,
# the Model Performance sub-tabs) loops over this instead of repeating
# the same three blocks of code with different filenames baked in.
TARGET_CONFIG = {
    "points_finish": {
        "label": "Points finish",
        "description": "Finishes in the top 10",
        "model_file": "final_model.joblib",
        "metadata_file": "model_metadata.json",
        "comparison_file": "model_comparison.csv",
        "is_regression": False,
    },
    "on_podium": {
        "label": "Podium",
        "description": "Finishes in the top 3",
        "model_file": "podium_model.joblib",
        "metadata_file": "podium_metadata.json",
        "comparison_file": "podium_model_comparison.csv",
        "is_regression": False,
    },
    "finish_position": {
        "label": "Finish position",
        "description": "Predicted finishing spot, 1st to 20th",
        "model_file": "finish_position_model.joblib",
        "metadata_file": "finish_position_metadata.json",
        "comparison_file": "finish_position_model_comparison.csv",
        "is_regression": True,
    },
}


# ═══════════════════════════════════════════════════════════════════
# WHAT: Loads a fitted model pipeline saved by the notebook.
# WHY: `st.cache_resource` keeps it loaded once per session instead of
#      re-reading the file from disk on every click in the app. Takes
#      a filename instead of being hardcoded so the same function
#      loads all three models.
# INPUT: filename, the .joblib file's name inside models/
# PLAN:
#   1. Read the joblib file the notebook saved.
# OUTPUT: a fitted scikit-learn Pipeline (preprocessing + estimator).
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model(filename):
    return joblib.load(MODELS_DIR / filename)


# ═══════════════════════════════════════════════════════════════════
# WHAT: Loads a metadata file saved alongside a model.
# WHY: The dashboard needs to know each model's exact feature order,
#      the country codes it was trained on, and its test-set scores,
#      without hardcoding any of it -- if a notebook re-run changes
#      the feature list or the numbers, this file changes too, and
#      the dashboard follows automatically.
# INPUT: filename, the .json file's name inside models/
# PLAN:
#   1. Read and parse the metadata JSON.
# OUTPUT: a dict with model_name, feature lists, and test-set scores.
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_metadata(filename):
    with open(MODELS_DIR / filename) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
# WHAT: Loads a model-comparison table saved by the notebook.
# WHY: Same reasoning as load_model -- this dashboard shows what the
#      notebook already found, it doesn't recompute it.
# INPUT: filename, the .csv file's name inside models/
# PLAN:
#   1. Read the comparison CSV.
# OUTPUT: a small DataFrame, one row per model tried for that target.
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_comparison_table(filename):
    return pd.read_csv(MODELS_DIR / filename)


# Columns that are genuinely track-specific, not driver-specific.
# `country_code` is deliberately not one of them: in this dataset it's
# driver nationality, not the race's country, so it stays tied to
# whichever driver is selected, not to the track.
CIRCUIT_PROFILE_COLUMNS = [
    "best_qual_time_s", "air_temp_max", "track_temp_min",
    "track_temp_range", "wind_speed_mean", "winddirection_mean",
]


# ═══════════════════════════════════════════════════════════════════
# WHAT: For every driver, loads their single most recent race row.
# WHY: Powers Simple mode in the Predict tab -- instead of a user typing
#      22 abstract numbers, they pick a real driver by name and this
#      supplies their actual, real, most recent form/standing/team
#      values. Built and saved by `04_model.ipynb` (Section 15), not
#      computed here, same rule as every other loader in this file.
# INPUT: none
# PLAN:
#   1. Read the lookup table the notebook saved.
# OUTPUT: one row per driver, columns = name/team/circuit/round + features
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_driver_lookup():
    return pd.read_csv(MODELS_DIR / "driver_lookup.csv")


# ═══════════════════════════════════════════════════════════════════
# WHAT: One row per circuit, averaging its qualifying pace and weather
#       across every real race held there.
# WHY: Powers the track picker in Simple mode. Picking a track changes
#      what a "typical" qualifying time and weather day look like a lot
#      (Monaco vs. Spa), so the input has to change with it, using real
#      historical numbers rather than a single fixed default for every
#      track. Built and saved by `04_model.ipynb` (Section 15).
# INPUT: none
# PLAN:
#   1. Read the circuit profile table the notebook saved.
# OUTPUT: a DataFrame, one row per circuit, circuit_name + its averages
# ═══════════════════════════════════════════════════════════════════
@st.cache_data
def load_circuit_profiles():
    return pd.read_csv(MODELS_DIR / "circuit_profiles.csv")


st.set_page_config(
    page_title="F1 Race Outcome Predictor",
    page_icon="🏎️",
    layout="wide",
)

# A light touch of custom styling on top of Streamlit's own theme: a
# little breathing room above the title, and metric numbers large enough
# to actually anchor each card instead of getting lost next to their
# label. Everything else (card borders, colours, dark/light mode) comes
# from Streamlit's native container(border=True) and its own theme, not
# from CSS overrides, so it keeps working if the user switches themes.
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.7rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Loads every model, its metadata, and its comparison table once, keyed
# by target id, so the rest of the file can just do loaded["on_podium"]
# instead of three separate sets of variables.
loaded = {}
for target_id, config in TARGET_CONFIG.items():
    loaded[target_id] = {
        "model": load_model(config["model_file"]),
        "metadata": load_metadata(config["metadata_file"]),
        "comparison": load_comparison_table(config["comparison_file"]),
    }

points_metadata = loaded["points_finish"]["metadata"]

st.title("🏎️ F1 Race Outcome Predictor")
st.markdown(
    "Three models, one shared set of pre-race signals: grid position, "
    "qualifying pace, weather, and recent form. No information from the "
    "race itself goes into any of them."
)

st.markdown("#### At a glance")
glance_cols = st.columns(3)
for glance_col, (target_id, config) in zip(glance_cols, TARGET_CONFIG.items()):
    target_metadata = loaded[target_id]["metadata"]
    with glance_col:
        with st.container(border=True):
            st.caption(config["description"])
            if config["is_regression"]:
                st.metric("Mean error", f"{target_metadata['test_mae']:.2f} positions")
            else:
                st.metric("F1 score", f"{target_metadata['test_f1']:.3f}")
            st.caption(f"Best model: {target_metadata['model_name']}")

insights_tab, predict_tab, performance_tab = st.tabs(
    ["🏁 Insights", "🔮 Predict", "📊 Model Performance"]
)


# ── Insights tab ─────────────────────────────────────────────────────
# Every caption below states a real, computed number from the EDA
# notebook -- nothing here is invented for the sake of a nice-sounding
# caption.
with insights_tab:
    st.subheader("What the EDA found")

    insights = [
        (
            "eda_04_grid_vs_finish.png",
            "Grid position vs. finishing position (r = 0.61). Starting near "
            "the front matters a lot: pole sitters in this dataset go on to "
            "win 55% of the time.",
        ),
        (
            "eda_09_avg_finish_by_constructor.png",
            "Average finishing position by constructor. Red Bull (5.8) and "
            "Sauber (14.6) sit almost 9 full positions apart on average. "
            "The car matters as much as the driver.",
        ),
        (
            "eda_08_constructor_points_by_season.png",
            "Constructor points by season. Red Bull's steady dominance and "
            "McLaren's climb back to the front are both visible year over "
            "year.",
        ),
        (
            "eda_12_five_stop_anomaly.png",
            "The '5 pit stops' group looks like a good strategy on average, "
            "but it isn't: those rows mostly come from 3 chaotic wet "
            "races, not a deliberate 5-stop strategy.",
        ),
        (
            "eda_13_pit_strategy_vs_finish.png",
            "Team pit stop *duration* correlates with results (r = 0.69, "
            "p = 0.014); how many stops a team takes, and when, barely "
            "does. Faster pit crews cluster with faster teams overall.",
        ),
        (
            "eda_17_final_prerace_correlation_heatmap.png",
            "The final pre-race feature correlation check. This is what "
            "decided the original 8 race-day features (notebook 02, Section "
            "12). 4 more, rolling driver and constructor form, were added "
            "later in notebook 03.",
        ),
    ]

    left_col, right_col = st.columns(2)
    for i, (filename, caption) in enumerate(insights):
        target_col = left_col if i % 2 == 0 else right_col
        image_path = FIGURES_DIR / filename
        if image_path.exists():
            with target_col:
                with st.container(border=True):
                    st.image(str(image_path), use_container_width=True)
                    st.caption(caption)


# ── Predict tab ──────────────────────────────────────────────────────
with predict_tab:
    st.subheader("Try a prediction")

    mode = st.radio(
        "Mode",
        ["Simple: pick a driver", "Advanced: set every input"],
        horizontal=True,
        label_visibility="collapsed",
    )

    simple_submitted = False
    advanced_submitted = False

    if mode == "Simple: pick a driver":
        st.markdown(
            "Pick a real driver and a track. The driver's recent form, "
            "team, and championship standing come from their actual most "
            "recent race; the track sets qualifying pace and weather to "
            "that circuit's real historical average. Real numbers "
            "throughout, nothing invented. Only grid position is yours to "
            "set, since that's specific to a hypothetical race and can't "
            "be known ahead of time."
        )

        driver_rows = load_driver_lookup()
        circuit_profiles = load_circuit_profiles()

        with st.form("simple_prediction_form"):
            driver_col, track_col = st.columns(2)

            with driver_col:
                selected_driver = st.selectbox(
                    "Driver", options=driver_rows["driver_name"].tolist(),
                )
                driver_row_df = driver_rows[driver_rows["driver_name"] == selected_driver]
                constructor_name = driver_row_df["constructor_name"].iloc[0]
                last_circuit = driver_row_df["circuit_name"].iloc[0]
                last_round = int(driver_row_df["round"].iloc[0])
                default_grid = int(driver_row_df["grid_position"].iloc[0])

                st.caption(
                    f"Team: {constructor_name}. Form data is from their "
                    f"most recent race, round {last_round} at {last_circuit}."
                )

            with track_col:
                circuit_options = circuit_profiles["circuit_name"].tolist()
                # Defaults to the driver's own most recent circuit, so
                # picking a driver alone (without touching the track)
                # still gives a self-consistent scenario.
                default_track_index = circuit_options.index(last_circuit) if last_circuit in circuit_options else 0
                selected_track = st.selectbox(
                    "Track", options=circuit_options, index=default_track_index,
                )
                track_row_df = circuit_profiles[circuit_profiles["circuit_name"] == selected_track]

                st.caption(
                    "Qualifying pace and weather are filled in from this "
                    "circuit's real historical average across every race "
                    "held there in the dataset."
                )

            simple_grid_position = st.number_input(
                "Grid position for this race",
                min_value=0, max_value=20, value=default_grid, step=1,
                help="0 = pit lane start. Defaults to the driver's actual "
                     "grid position last time out.",
            )

            simple_submitted = st.form_submit_button("Predict")

        if simple_submitted:
            input_row = driver_row_df[points_metadata["all_features"]].copy()
            input_row["grid_position"] = simple_grid_position
            # Overwrites the track-specific columns with the selected
            # circuit's real averages -- country_code is left untouched,
            # since it's the driver's nationality, not the race's
            # location, and has nothing to do with which track this is.
            for column in CIRCUIT_PROFILE_COLUMNS:
                input_row[column] = track_row_df[column].iloc[0]

            # float(...) matters here: predict_proba/predict return numpy
            # float32, and st.progress (used below) rejects anything that
            # isn't a plain Python float or int.
            st.session_state["predictions"] = {
                "points_proba": float(loaded["points_finish"]["model"].predict_proba(input_row)[0, 1]),
                "podium_proba": float(loaded["on_podium"]["model"].predict_proba(input_row)[0, 1]),
                "position_raw": float(loaded["finish_position"]["model"].predict(input_row)[0]),
                "driver_name": selected_driver,
                "constructor_name": constructor_name,
            }

    else:
        st.markdown(
            f"Set every one of the {len(points_metadata['all_features'])} "
            f"features by hand. Useful for testing an exact scenario, or "
            f"seeing how far each model's prediction moves when one input "
            f"changes."
        )

        with st.form("advanced_prediction_form"):
            race_col, weather_col, driver_col, constructor_col = st.columns(4)

            with race_col:
                st.markdown("**Race setup**")
                grid_position = st.number_input(
                    "Grid position", min_value=0, max_value=20, value=10, step=1,
                    help="0 = pit lane start",
                )
                best_qual_time_s = st.number_input(
                    "Best qualifying time (seconds)",
                    min_value=60.0, max_value=130.0, value=84.5, step=0.1,
                )
                season_progress = st.slider(
                    "Season progress",
                    min_value=0.0, max_value=1.0, value=0.5,
                    help="0 = first race of the season, 1 = last",
                )
                country_code = st.selectbox(
                    "Country", options=points_metadata["country_code_options"],
                )

            with weather_col:
                st.markdown("**Weather**")
                air_temp_max = st.number_input(
                    "Max air temperature (°C)",
                    min_value=10.0, max_value=40.0, value=25.6, step=0.5,
                )
                track_temp_min = st.number_input(
                    "Min track temperature (°C)",
                    min_value=10.0, max_value=50.0, value=31.3, step=0.5,
                )
                track_temp_range = st.number_input(
                    "Track temperature range (°C, max - min)",
                    min_value=0.0, max_value=25.0, value=6.3, step=0.5,
                )
                wind_speed_mean = st.number_input(
                    "Average wind speed (m/s)",
                    min_value=0.0, max_value=6.0, value=1.6, step=0.1,
                )
                winddirection_mean = st.slider(
                    "Average wind direction (degrees)",
                    min_value=0.0, max_value=360.0, value=184.0,
                )

            with driver_col:
                st.markdown("**Driver, last 5 races**")
                driver_form_points_last5 = st.number_input(
                    "Driver's average points",
                    min_value=0.0, max_value=26.0, value=2.2, step=0.5,
                )
                driver_points_finish_rate_last5 = st.slider(
                    "Driver's points-finish rate",
                    min_value=0.0, max_value=1.0, value=0.5,
                    help="Fraction of the last 5 races the driver scored points in",
                )
                driver_dnf_rate_last5 = st.slider(
                    "Driver's DNF rate",
                    min_value=0.0, max_value=1.0, value=0.4,
                    help="Fraction of the last 5 races the driver didn't finish",
                )
                driver_grid_position_last5 = st.number_input(
                    "Driver's average grid position",
                    min_value=1.0, max_value=20.0, value=10.8, step=0.5,
                )
                driver_positions_gained_last5 = st.slider(
                    "Driver's positions gained",
                    min_value=-15.0, max_value=10.0, value=0.2, step=0.5,
                    help="Grid position minus finish position, averaged. Positive means gaining places.",
                )
                driver_pace_ratio_last5 = st.number_input(
                    "Driver's pace ratio",
                    min_value=0.85, max_value=1.15, value=1.00, step=0.01,
                    help="This driver's lap time divided by the field median, averaged across races so it's comparable circuit to circuit. Below 1.0 means faster than the field.",
                )

                st.markdown("**Driver, this season**")
                driver_teammate_gap_last5 = st.slider(
                    "Gap to teammate",
                    min_value=-20.0, max_value=20.0, value=0.0, step=0.5,
                    help="Average finishing-position gap to the team-mate, last 5 races. Positive means finishing behind them.",
                )
                driver_season_avg_points_so_far = st.number_input(
                    "Driver's season average points",
                    min_value=0.0, max_value=26.0, value=2.0, step=0.5,
                )
                driver_standing_before_race = st.number_input(
                    "Driver's championship standing", min_value=1, max_value=20, value=10, step=1,
                    help="1 = leading the drivers' championship going into this race",
                )

            with constructor_col:
                st.markdown("**Constructor, last 5 races**")
                constructor_form_points_last5 = st.number_input(
                    "Constructor's average points",
                    min_value=0.0, max_value=26.0, value=2.4, step=0.5,
                    help="Average points scored by either car from this team",
                )
                constructor_dnf_rate_last5 = st.slider(
                    "Constructor's DNF rate",
                    min_value=0.0, max_value=1.0, value=0.4,
                    help="Fraction of the last 5 races either car didn't finish",
                )
                constructor_avg_pit_duration_last5 = st.number_input(
                    "Average pit stop duration (s)",
                    min_value=18.0, max_value=36.0, value=24.4, step=0.5,
                )

                st.markdown("**Constructor, this season**")
                constructor_standing_before_race = st.number_input(
                    "Constructor's championship standing", min_value=1, max_value=20, value=9, step=1,
                    help="1 = leading the constructors' championship going into this race",
                )
                constructor_points_finish_rate = st.slider(
                    "Constructor's points-finish rate",
                    min_value=0.0, max_value=1.0, value=0.51,
                    help="Fraction of this team's races, across the whole dataset, that ended in a points finish",
                )

            advanced_submitted = st.form_submit_button("Predict")

        if advanced_submitted:
            # Build a single-row DataFrame in the exact column order the models
            # were trained on -- every saved pipeline handles scaling/encoding
            # internally, so raw values go in as-is. All three models share the
            # same feature schema, so the same row feeds all three.
            input_row = pd.DataFrame([{
                "grid_position": grid_position,
                "best_qual_time_s": best_qual_time_s,
                "air_temp_max": air_temp_max,
                "track_temp_min": track_temp_min,
                "wind_speed_mean": wind_speed_mean,
                "winddirection_mean": winddirection_mean,
                "track_temp_range": track_temp_range,
                "season_progress": season_progress,
                "driver_form_points_last5": driver_form_points_last5,
                "driver_points_finish_rate_last5": driver_points_finish_rate_last5,
                "driver_dnf_rate_last5": driver_dnf_rate_last5,
                "driver_grid_position_last5": driver_grid_position_last5,
                "driver_teammate_gap_last5": driver_teammate_gap_last5,
                "driver_season_avg_points_so_far": driver_season_avg_points_so_far,
                "driver_positions_gained_last5": driver_positions_gained_last5,
                "driver_pace_ratio_last5": driver_pace_ratio_last5,
                "constructor_form_points_last5": constructor_form_points_last5,
                "constructor_dnf_rate_last5": constructor_dnf_rate_last5,
                "constructor_avg_pit_duration_last5": constructor_avg_pit_duration_last5,
                "driver_standing_before_race": driver_standing_before_race,
                "constructor_standing_before_race": constructor_standing_before_race,
                "constructor_points_finish_rate": constructor_points_finish_rate,
                "country_code": country_code,
            }])[points_metadata["all_features"]]

            # float(...) matters here: predict_proba/predict return numpy
            # float32, and st.progress (used below) rejects anything that
            # isn't a plain Python float or int.
            st.session_state["predictions"] = {
                "points_proba": float(loaded["points_finish"]["model"].predict_proba(input_row)[0, 1]),
                "podium_proba": float(loaded["on_podium"]["model"].predict_proba(input_row)[0, 1]),
                "position_raw": float(loaded["finish_position"]["model"].predict(input_row)[0]),
                "driver_name": None,
                "constructor_name": None,
            }

    # Stashed in session_state (survives the rerun the threshold slider
    # below triggers) so moving the slider updates the points-finish card
    # instantly without re-running any model on the form inputs again.
    if "predictions" in st.session_state:
        predictions = st.session_state["predictions"]

        if predictions["driver_name"]:
            st.markdown(f"#### Results for {predictions['driver_name']} ({predictions['constructor_name']})")
        else:
            st.markdown("#### Results")

        result_cols = st.columns(3)

        with result_cols[0]:
            with st.container(border=True):
                st.markdown("**Points finish**")
                st.caption("Finishes in the top 10")

                threshold = st.slider(
                    "Decision threshold",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                    help=(
                        "Not hardcoded, move it and watch the prediction "
                        "change. Lower catches more real points finishes "
                        "but is wrong more often when it says yes; higher "
                        "is pickier but misses more. On the 2025 test set, "
                        "0.27 was the threshold that maximised F1 (0.751 "
                        "vs. 0.739 at the default 0.50), trading 9 points "
                        "of precision for 13 points of recall to get "
                        "there. It isn't set as the default here on "
                        "purpose, since that tradeoff is a call "
                        "about what the prediction is for, not something "
                        "a metric decides by itself."
                    ),
                )
                points_label = "Points finish" if predictions["points_proba"] >= threshold else "No points"

                st.metric("Prediction", points_label)
                st.metric("Probability", f"{predictions['points_proba']:.1%}")
                st.progress(predictions["points_proba"])

        with result_cols[1]:
            with st.container(border=True):
                st.markdown("**Podium**")
                st.caption("Finishes in the top 3")

                podium_label = "Podium" if predictions["podium_proba"] >= 0.5 else "No podium"

                st.metric("Prediction", podium_label)
                st.metric("Probability", f"{predictions['podium_proba']:.1%}")
                st.progress(predictions["podium_proba"])
                st.caption(
                    "Only 15% of driver-races actually reach the podium, "
                    "so treat a close call here as meaningful even below "
                    "50%."
                )

        with result_cols[2]:
            with st.container(border=True):
                st.markdown("**Finish position**")
                st.caption("Predicted finishing spot, 1st to 20th")

                position_raw = predictions["position_raw"]
                position_clipped = min(max(position_raw, 1), 20)
                position_mae = loaded["finish_position"]["metadata"]["test_mae"]

                st.metric("Predicted position", f"P{round(position_clipped)}")
                st.caption(
                    f"Model's raw estimate: {position_raw:.1f}. Typically "
                    f"off by about {position_mae:.1f} positions on the "
                    f"2025 test set, so read this as a rough zone, not an "
                    f"exact call."
                )


# ── Model Performance tab ────────────────────────────────────────────
with performance_tab:
    st.subheader("How each model was chosen")
    st.markdown(
        "All models were trained on 2022 through 2024 and evaluated on "
        "the full 2025 season, which they never saw during training. "
        "That's a time-based split, not a random one, since race results "
        "aren't independent of each other."
    )

    with st.container(border=True):
        st.markdown("**Why did a different model win for each target?**")
        st.markdown(
            "It isn't the same algorithm every time, and that's not "
            "random. Each target has a different shape, and the winner "
            "is whichever model's strengths match that shape.\n\n"
            "**Points finish (near 50/50 split): XGBoost wins on F1, "
            "but Logistic Regression still has the better ROC-AUC.** "
            "Several features here move together (a driver's recent "
            "points, their team's recent points, and grid position are "
            "all correlated). A linear model has to assign each one its "
            "own fixed weight, so correlated inputs make its decision "
            "right at the 50% cutoff less stable. A tree-based model "
            "like XGBoost sidesteps this: at each split it just picks "
            "whichever feature is most useful *right now*, so redundant "
            "features stop mattering once a better one has already been "
            "used. That's why XGBoost edges out F1, while Logistic "
            "Regression's smoother probability estimates still rank "
            "drivers slightly better overall (ROC-AUC).\n\n"
            "**Podium (rare event, only 15% True): XGBoost wins by a "
            "wide margin.** Reaching the podium likely isn't one thing "
            "mattering on its own, it's several things needing to line "
            "up together: strong recent form *and* a good grid slot "
            "*and* a reliable car. A linear model can only add up each "
            "feature's effect separately, it can't represent \"this "
            "matters more when that other thing is also true.\" A "
            "tree-based model captures exactly that kind of AND "
            "condition naturally, which matters even more once the "
            "positive class is this rare.\n\n"
            "**Finish position (regression, continuous 1-20): plain "
            "Linear Regression wins, only just, over Random Forest and "
            "XGBoost.** This is the noisiest target of the three (R² "
            "tops out around 0.44), a lot of what decides an exact "
            "finishing position (a lap-1 crash, a mid-race mechanical "
            "failure) simply isn't knowable before the race. When there "
            "isn't much learnable signal left, a more flexible model "
            "has more room to fit noise in the training data instead of "
            "real patterns, and that costs it on the held-out test set. "
            "The simpler model doesn't have that extra flexibility to "
            "misuse, so it holds up just as well, or slightly better.\n\n"
            "The common thread: model choice depends on how balanced "
            "the target is, whether the real relationship needs "
            "interactions a linear model can't express, and how much "
            "learnable signal actually exists versus noise. That's the "
            "real reason this project ran a full model comparison "
            "separately for every target instead of assuming one "
            "algorithm would win everywhere."
        )

    points_perf_tab, podium_perf_tab, position_perf_tab = st.tabs(
        ["Points finish", "Podium", "Finish position"]
    )

    with points_perf_tab:
        st.dataframe(loaded["points_finish"]["comparison"], use_container_width=True)

        chart_col1, chart_col2 = st.columns(2)
        roc_path = FIGURES_DIR / "model_02_roc_comparison.png"
        confusion_path = FIGURES_DIR / "model_03_best_confusion_matrix.png"
        if roc_path.exists():
            chart_col1.image(str(roc_path), use_container_width=True)
            chart_col1.caption(
                "ROC curves for all 4 models. The dummy classifier's curve "
                "is the literal diagonal random-guessing line. Every real "
                "model bowing above it is proof it learned something."
            )
        if confusion_path.exists():
            chart_col2.image(str(confusion_path), use_container_width=True)
            chart_col2.caption(
                f"Confusion matrix for the winning model, "
                f"{loaded['points_finish']['metadata']['model_name']}."
            )

        importance_path = FIGURES_DIR / "model_04_feature_importance.png"
        if importance_path.exists():
            st.image(str(importance_path), use_container_width=True)
            st.caption(
                "What the winning model actually leaned on. One honest "
                "caveat: country columns and numeric features aren't on "
                "the same scale, so their bar sizes aren't strictly "
                "comparable."
            )

    with podium_perf_tab:
        st.dataframe(loaded["on_podium"]["comparison"], use_container_width=True)
        st.caption(
            "Only 15% of driver-races end on the podium, far more "
            "lopsided than the points-finish target's roughly even split. "
            "Worth comparing the dummy row's F1 against its accuracy here: "
            "high accuracy, zero F1, a clean example of why accuracy alone "
            "can be misleading on an imbalanced target."
        )

        podium_confusion_path = FIGURES_DIR / "model_06_podium_confusion_matrix.png"
        if podium_confusion_path.exists():
            st.image(str(podium_confusion_path), use_container_width=True)
            st.caption(
                f"Confusion matrix for the winning model, "
                f"{loaded['on_podium']['metadata']['model_name']}."
            )

    with position_perf_tab:
        st.dataframe(loaded["finish_position"]["comparison"], use_container_width=True)
        st.caption(
            "Lower is better for MAE and RMSE, higher is better for R2. "
            "This target is a genuinely harder problem than the other "
            "two: predicting an exact finishing position has to account "
            "for things no pre-race feature can see, like a lap-1 crash "
            "or a mid-race mechanical failure."
        )

        position_scatter_path = FIGURES_DIR / "model_07_finish_position_predicted_vs_actual.png"
        if position_scatter_path.exists():
            st.image(str(position_scatter_path), use_container_width=True)
            st.caption(
                "Predicted vs. actual finish position on the 2025 test "
                "set. Predictions cluster toward the middle of the grid "
                "even for drivers who actually finished 1st or 20th, a "
                "common regression tendency, not a bug."
            )
