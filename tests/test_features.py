"""Feature engineering tests।"""

import pandas as pd
import pytest

from src.features.feature_engineering import (
    add_power_features,
    add_temperature_features,
    add_wear_features,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "air_temperature": [298.0, 300.0],
            "process_temperature": [308.0, 310.0],
            "rotational_speed": [1500.0, 2000.0],
            "torque": [40.0, 55.0],
            "tool_wear": [50.0, 100.0],
        }
    )


def test_temperature_features_added(sample_df):
    result = add_temperature_features(sample_df)
    assert "temp_diff" in result.columns
    assert "temp_ratio" in result.columns


def test_temp_diff_calculation(sample_df):
    result = add_temperature_features(sample_df)
    expected = sample_df["process_temperature"] - sample_df["air_temperature"]
    pd.testing.assert_series_equal(result["temp_diff"], expected, check_names=False)


def test_power_features_added(sample_df):
    result = add_power_features(sample_df)
    assert "power_consumption" in result.columns
    assert "power_wear_interaction" in result.columns


def test_no_nan_in_features(sample_df):
    result = add_temperature_features(sample_df)
    result = add_power_features(result)
    result = add_wear_features(result)
    assert result.isnull().sum().sum() == 0
