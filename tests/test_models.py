import math

import pandas as pd
import pytest

import modules.cnn_model as cnn_module
import modules.fuzzy_logic as fuzzy_module
import modules.mlr as mlr_module
from modules.config import FUZZY_OUTPUT_MAP
from modules.exceptions import InvalidInputError
from modules.feature_extraction import FEATURE_NAMES, extract_features_from_df


@pytest.fixture
def features(vibration_df):
    return extract_features_from_df(vibration_df)


def test_mlr_artifacts_present_and_predict_returns_finite_float(features):
    assert mlr_module.is_available()
    ra = mlr_module.predict(features)
    assert isinstance(ra, float) and math.isfinite(ra)


def test_mlr_equation_names_its_four_features():
    names = mlr_module.get_feature_names()
    assert len(names) == 4
    assert all(n in FEATURE_NAMES for n in names)
    equation = mlr_module.get_equation()
    assert equation.startswith("Ra =")
    assert all(n in equation for n in names)


def test_mlr_rejects_input_missing_its_features():
    with pytest.raises(InvalidInputError):
        mlr_module.predict(pd.DataFrame([{"irrelevant": 1.0}]))


def test_fuzzy_prediction_stays_within_physical_bounds(features):
    ra = fuzzy_module.predict(features)
    assert FUZZY_OUTPUT_MAP["Smooth"] <= ra <= FUZZY_OUTPUT_MAP["Rough"]


def test_fuzzy_uses_all_81_rules():
    assert fuzzy_module.get_rule_count() == 81


@pytest.mark.parametrize(
    "ra,expected",
    [(1.5, "Smooth"), (2.0, "Smooth"), (2.3, "Average"), (2.5, "Average"), (2.8, "Rough")],
)
def test_fuzzy_categories_match_thresholds(ra, expected):
    assert fuzzy_module.categorize(ra) == expected


def test_fuzzy_rejects_input_missing_channel_2_features():
    with pytest.raises(InvalidInputError):
        fuzzy_module.predict(pd.DataFrame([{"Channel 1 RMS": 1.0}]))


@pytest.mark.skipif(not cnn_module.is_available(), reason="TensorFlow or CNN artifacts missing")
def test_cnn_predict_returns_finite_float(features):
    ra = cnn_module.predict(features)
    assert isinstance(ra, float) and math.isfinite(ra)


@pytest.mark.skipif(not cnn_module.is_available(), reason="TensorFlow or CNN artifacts missing")
def test_cnn_rejects_input_missing_features():
    with pytest.raises(InvalidInputError):
        cnn_module.predict(pd.DataFrame([{"Channel 1 RMS": 1.0}]))


def test_predictions_are_deterministic(features):
    assert mlr_module.predict(features) == mlr_module.predict(features)
    assert fuzzy_module.predict(features) == fuzzy_module.predict(features)
