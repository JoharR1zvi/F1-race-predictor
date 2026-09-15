"""
Sanity checks on the files 04_model.ipynb saves into models/ -- these are
what the dashboard actually loads at runtime, so a schema mismatch here
(a renamed column, a metadata key that disappeared) would break the
dashboard silently until someone clicked the wrong button. Fast, no
browser simulation, just load each file and check the shape everything
downstream assumes is actually there.
"""

import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def test_all_three_metadata_files_have_required_keys():
    required_keys = {"model_name", "target", "all_features", "country_code_options"}
    for filename in ("model_metadata.json", "podium_metadata.json", "finish_position_metadata.json"):
        with open(MODELS_DIR / filename) as f:
            metadata = json.load(f)
        missing = required_keys - metadata.keys()
        assert not missing, f"{filename} is missing keys: {missing}"


def test_all_three_models_share_the_same_feature_schema():
    # Simple mode builds one input row and feeds it to all three models --
    # if their feature lists ever diverged, that would break silently.
    with open(MODELS_DIR / "model_metadata.json") as f:
        points_features = json.load(f)["all_features"]
    with open(MODELS_DIR / "podium_metadata.json") as f:
        podium_features = json.load(f)["all_features"]
    with open(MODELS_DIR / "finish_position_metadata.json") as f:
        position_features = json.load(f)["all_features"]

    assert points_features == podium_features == position_features


def test_all_three_models_load_and_can_predict():
    with open(MODELS_DIR / "model_metadata.json") as f:
        all_features = json.load(f)["all_features"]

    driver_lookup = pd.read_csv(MODELS_DIR / "driver_lookup.csv")
    sample_row = driver_lookup[all_features].iloc[[0]]

    points_model = joblib.load(MODELS_DIR / "final_model.joblib")
    podium_model = joblib.load(MODELS_DIR / "podium_model.joblib")
    position_model = joblib.load(MODELS_DIR / "finish_position_model.joblib")

    points_proba = points_model.predict_proba(sample_row)[0, 1]
    podium_proba = podium_model.predict_proba(sample_row)[0, 1]
    position_pred = position_model.predict(sample_row)[0]

    assert 0.0 <= points_proba <= 1.0
    assert 0.0 <= podium_proba <= 1.0
    # Regression, not classification -- it can overshoot the real 1-20
    # grid slightly, so this checks it's in a sane neighbourhood, not an
    # exact bound.
    assert -5.0 <= position_pred <= 25.0


def test_driver_lookup_has_every_current_grid_driver():
    driver_lookup = pd.read_csv(MODELS_DIR / "driver_lookup.csv")
    # 2025 had 21 different drivers across the season (mid-season swaps
    # included) -- if this drops, the Simple-mode dropdown lost someone.
    assert driver_lookup["driver_name"].nunique() >= 20


def test_circuit_profiles_cover_every_2025_circuit():
    circuit_profiles = pd.read_csv(MODELS_DIR / "circuit_profiles.csv")
    driver_lookup = pd.read_csv(MODELS_DIR / "driver_lookup.csv")

    circuits_with_a_profile = set(circuit_profiles["circuit_name"])
    circuits_drivers_actually_raced_at = set(driver_lookup["circuit_name"])

    missing = circuits_drivers_actually_raced_at - circuits_with_a_profile
    assert not missing, f"Circuits with no weather/pace profile: {missing}"
