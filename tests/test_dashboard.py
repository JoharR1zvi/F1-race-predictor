"""
Tests for app/dashboard.py using Streamlit's own AppTest framework -- it
runs the real app script in a simulated session, without a browser or a
running server, and lets a test click buttons and read back what got
rendered.

These exist because two real bugs slipped past manual checking earlier
in this project: a missing form field that only crashed once someone
actually clicked Predict, and a numpy-vs-Python-float type mismatch that
only crashed once st.progress() actually ran on a real value. A "does
the page load" check missed both -- these tests click Predict for real,
in both modes, which is the only way that class of bug gets caught
before a user finds it.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

DASHBOARD_PATH = str(Path(__file__).resolve().parent.parent / "app" / "dashboard.py")


def test_dashboard_loads_without_error():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    assert not at.exception


def test_simple_mode_predict_runs_without_error():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)
    assert not at.exception

    # Simple mode is the default radio selection -- just submit the form
    # with whatever driver/track/grid-position defaults it starts with.
    at.button[0].click().run(timeout=30)

    assert not at.exception
    # A results header only appears after a prediction actually completed.
    assert any("Results" in md.value for md in at.markdown)


def test_advanced_mode_predict_runs_without_error():
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    at.radio[0].set_value("Advanced: set every input").run(timeout=30)
    assert not at.exception

    at.button[0].click().run(timeout=30)

    assert not at.exception
    assert any("Results" in md.value for md in at.markdown)


def test_track_choice_changes_the_prediction():
    # Regression test for the track picker specifically: picking a
    # different circuit should change the weather/qualifying-pace inputs
    # the model sees, and therefore change its output. If this ever
    # starts failing, the track selection has stopped actually reaching
    # the model.
    at = AppTest.from_file(DASHBOARD_PATH)
    at.run(timeout=30)

    # Selectboxes in the Simple-mode form, in the order they're created:
    # [0] driver, [1] track.
    at.selectbox[1].set_value("Circuit Paul Ricard").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    monaco_probability = at.session_state["predictions"]["points_proba"]

    at.selectbox[1].set_value("Circuit de Monaco").run(timeout=30)
    at.button[0].click().run(timeout=30)
    assert not at.exception
    ricard_probability = at.session_state["predictions"]["points_proba"]

    assert monaco_probability != ricard_probability
