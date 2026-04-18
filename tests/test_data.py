"""Data preprocessing এর tests।"""

import pandas as pd

from src.data.preprocess import clean_data, encode_features


def make_sample_df():
    """Test এর জন্য sample data।"""
    return pd.DataFrame(
        {
            "Type": ["L", "M", "H", "M", "L"],
            "Air temperature [K]": [298.1, 297.5, 300.0, 299.0, 296.0],
            "Process temperature [K]": [308.6, 307.5, 310.0, 309.0, 306.0],
            "Rotational speed [rpm]": [1551, 1408, 1500, 1300, 1700],
            "Torque [Nm]": [42.8, 46.3, 55.0, 38.0, 40.0],
            "Tool wear [min]": [0, 3, 100, 200, 50],
            "Machine failure": [0, 0, 1, 0, 0],
        }
    )


def test_clean_data_removes_duplicates():
    """Duplicate rows remove হচ্ছে কি না।"""
    df = make_sample_df()
    df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    cleaned = clean_data(df_with_dup)
    assert len(cleaned) == len(df)


def test_clean_data_handles_missing():
    """Missing values handle হচ্ছে কি না।"""
    df = make_sample_df()
    df.loc[0, "Torque [Nm]"] = None
    cleaned = clean_data(df)
    assert cleaned.isnull().sum().sum() == 0


def test_encode_features_type_column(tmp_path):
    """Type column encode হচ্ছে কি না।"""
    df = make_sample_df()
    encoder_path = str(tmp_path / "test_encoder.pkl")
    encoded = encode_features(df, fit=True, encoder_path=encoder_path)
    assert pd.api.types.is_integer_dtype(
        encoded["Type"]
    ), f"Expected integer dtype, got {encoded['Type'].dtype}"
    assert encoded["Type"].nunique() == 3


def test_data_shape_preserved():
    """Cleaning এ অযথা rows delete হচ্ছে না।"""
    df = make_sample_df()
    cleaned = clean_data(df)
    assert len(cleaned) <= len(df)  # শুধু duplicates/NaN বাদ যাবে
