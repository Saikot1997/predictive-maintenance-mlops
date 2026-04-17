"""
Raw data clean এবং processed data তৈরি করে।
DVC pipeline এর প্রথম step।
"""
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(path: str = "configs/data_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Raw CSV load করো।"""
    logger.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Shape: {df.shape} | Columns: {list(df.columns)}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning।"""
    # Missing values check — fillna with median instead of dropping rows
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        # Fill numeric columns with median (preserves rows, better than dropna)
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include="object").columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        logger.info("Missing values filled with median/mode")

    # Duplicate rows remove
    dups = df.duplicated().sum()
    if dups > 0:
        logger.info(f"Removing {dups} duplicate rows")
        df = df.drop_duplicates()

    # Unnecessary columns drop (Product ID একটা identifier, ML তে লাগবে না)
    drop_cols = ["UDI", "Product ID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    logger.info(f"After cleaning shape: {df.shape}")
    return df


def encode_features(df: pd.DataFrame, fit: bool = True,
                    encoder_path: str = "models/trained/label_encoder.pkl") -> pd.DataFrame:
    """Categorical features encode করো।
    'Type' column: L/M/H → 0/1/2
    fit=True মানে নতুন encoder তৈরি করো (training)
    fit=False মানে existing encoder load করো (prediction)
    """
    df = df.copy()
    Path("models/trained").mkdir(parents=True, exist_ok=True)

    if fit:
        le = LabelEncoder()
        df["Type"] = le.fit_transform(df["Type"])
        joblib.dump(le, encoder_path)
        logger.info(f"Label encoder saved: {encoder_path}")
    else:
        le = joblib.load(encoder_path)
        df["Type"] = le.transform(df["Type"])

    return df


def scale_features(df: pd.DataFrame, feature_cols: list,
                   fit: bool = True,
                   scaler_path: str = "models/trained/scaler.pkl") -> pd.DataFrame:
    """Numeric features scale করো (StandardScaler: mean=0, std=1)।
    কেন scale করা লাগে: Rotational speed ৩০০০ RPM, Torque ৪০ Nm — এই দুটো খুব different scale।
    Model confuse হয়। Scale করলে সব same range এ আসে।
    """
    df = df.copy()
    Path("models/trained").mkdir(parents=True, exist_ok=True)

    if fit:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved: {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df


def run_preprocessing(config_path: str = "configs/data_config.yaml") -> None:
    """Full preprocessing pipeline চালাও।"""
    config = load_config(config_path)
    data_cfg = config["data"]

    # Load
    df = load_raw_data(data_cfg["raw_path"])

    # Clean
    df = clean_data(df)

    # Encode categorical (fit on full df before split — encoder sees all categories)
    df = encode_features(df, fit=True)

    # Rename columns — use clean names for everything downstream
    df = df.rename(columns={
        "Air temperature [K]": "air_temperature",
        "Process temperature [K]": "process_temperature",
        "Rotational speed [rpm]": "rotational_speed",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "target",
    })

    # ── STEP 1: Feature engineering on RAW (unscaled) values ──
    # MUST run before scaling so physics-based features are computed on
    # real sensor units (Kelvin, RPM, Nm, minutes) — not z-scores.
    from src.features.feature_engineering import (
        add_temperature_features, add_power_features, add_wear_features
    )
    df = add_temperature_features(df)
    df = add_power_features(df)
    df = add_wear_features(df)
    logger.info(f"After feature engineering: {df.shape[1]} columns")

    # ── STEP 2: Split BEFORE any scaling — prevents data leakage ──
    # failure_types keep their original names (TWF, HDF etc.)
    failure_cols = [c for c in data_cfg["failure_types"] if c in df.columns]
    X = df.drop(columns=["target"] + failure_cols)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y,                    # ← This line must be active
    )

    logger.info(f"Train failure rate: {y_train.mean():.4f} | Test failure rate: {y_test.mean():.4f}")
    logger.info(f"Unique classes in y_train: {np.unique(y_train)}")

    # ── STEP 3: Scale numeric features — fit on X_train ONLY ──
    # This is the correct order: no test-set statistics leak into the scaler.
    # predict.py receives raw sensor values and applies the same fitted scaler.
    numeric_features = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]
    X_train_scaled = scale_features(pd.DataFrame(X_train.values, columns=X_train.columns), numeric_features, fit=True)
    X_test_scaled  = scale_features(pd.DataFrame(X_test.values,  columns=X_test.columns),  numeric_features, fit=False)

    # Save
    Path(data_cfg["processed_train"]).parent.mkdir(parents=True, exist_ok=True)

    y_train_reset = y_train.reset_index(drop=True).astype(int)
    y_test_reset  = y_test.reset_index(drop=True).astype(int)

    train_df = X_train_scaled.copy()
    train_df["target"] = y_train_reset.values

    test_df = X_test_scaled.copy()
    test_df["target"] = y_test_reset.values

    train_df.to_parquet(data_cfg["processed_train"], index=False)
    test_df.to_parquet(data_cfg["processed_test"], index=False)

    logger.info(f"✅ Train: {train_df.shape}, Test: {test_df.shape}")
    logger.info(f"Train saved: {data_cfg['processed_train']}")
    logger.info(f"Test saved: {data_cfg['processed_test']}")

    # Class distribution দেখো
    logger.info(f"Target distribution:\n{y_train.value_counts(normalize=True)}")


if __name__ == "__main__":
    run_preprocessing()