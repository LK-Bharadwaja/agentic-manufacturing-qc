import numpy as np
import pandas as pd
import pytest

from modules.config import CHANNELS, SAMPLING_RATE, WINDOW_SIZE
from modules.exceptions import InvalidInputError, MissingChannelError
from modules.feature_extraction import (
    FEATURE_NAMES,
    available_channels,
    extract_features_from_df,
    extract_features_from_file,
)


def test_returns_36_features_in_declared_order(vibration_df):
    out = extract_features_from_df(vibration_df)
    assert out.shape == (1, 36)
    assert list(out.columns) == FEATURE_NAMES
    assert np.isfinite(out.to_numpy()).all()


def test_time_domain_matches_analytical_values(sine_signal):
    df = pd.DataFrame({CHANNELS[0]: sine_signal})
    out = extract_features_from_df(df)

    # Sampling is discrete, so the peak sample sits just below the true crest
    assert out["Channel 1 RMS"].iloc[0] == pytest.approx(2 / np.sqrt(2), rel=1e-3)
    assert out["Channel 1 PEAK"].iloc[0] == pytest.approx(2.0, rel=1e-2)
    assert out["Channel 1 CREST FACTOR"].iloc[0] == pytest.approx(np.sqrt(2), rel=1e-2)


def test_dominant_frequency_recovers_input_tone(sine_signal):
    df = pd.DataFrame({CHANNELS[0]: sine_signal})
    out = extract_features_from_df(df)
    # Bin width is SAMPLING_RATE / WINDOW_SIZE
    tolerance = SAMPLING_RATE / WINDOW_SIZE
    assert out["Channel 1 Dominant Frequency"].iloc[0] == pytest.approx(1000, abs=tolerance)


def test_missing_channels_are_zero_filled(vibration_df):
    partial = vibration_df[[CHANNELS[1]]]
    out = extract_features_from_df(partial)

    assert (out[[c for c in FEATURE_NAMES if c.startswith("Channel 1")]] == 0).all(axis=None)
    assert (out[[c for c in FEATURE_NAMES if c.startswith("Channel 3")]] == 0).all(axis=None)
    assert out["Channel 2 RMS"].iloc[0] > 0


def test_no_recognised_channel_is_rejected():
    with pytest.raises(MissingChannelError):
        extract_features_from_df(pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]}))


def test_short_signal_is_padded_not_dropped():
    short = pd.DataFrame({CHANNELS[0]: np.linspace(0, 1, WINDOW_SIZE // 4)})
    out = extract_features_from_df(short)
    assert out.shape == (1, 36)
    assert out["Channel 1 RMS"].iloc[0] > 0


def test_available_channels_reports_only_present_columns(vibration_df):
    assert available_channels(vibration_df) == CHANNELS
    assert available_channels(vibration_df[[CHANNELS[2]]]) == [CHANNELS[2]]
    assert available_channels(pd.DataFrame({"x": [1]})) == []


def test_unsupported_file_extension_is_rejected(tmp_path):
    path = tmp_path / "signal.txt"
    path.write_text("not a spreadsheet")
    with pytest.raises(InvalidInputError):
        extract_features_from_file(str(path))


def test_round_trip_through_csv_file(tmp_path, vibration_df):
    path = tmp_path / "part.csv"
    vibration_df.to_csv(path, index=False)
    out = extract_features_from_file(str(path))
    assert list(out.columns) == FEATURE_NAMES
