"""
নতুন features তৈরি করো — model এর performance বাড়ানোর জন্য।
Domain knowledge ব্যবহার করে meaningful features create করা।
"""
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """Temperature related নতুন features।"""
    df = df.copy()
    # Temperature difference — high difference = cooling problem
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    # Temperature ratio
    df["temp_ratio"] = df["process_temperature"] / (df["air_temperature"] + 1e-8)
    return df


def add_power_features(df: pd.DataFrame) -> pd.DataFrame:
    """Power consumption related features।
    Physics: Power = Torque × Angular Velocity (rpm × 2π/60)
    """
    df = df.copy()
    df["power_consumption"] = df["torque"] * df["rotational_speed"] * (2 * np.pi / 60)
    # High power + high tool wear = failure risk
    df["power_wear_interaction"] = df["power_consumption"] * df["tool_wear"]
    return df


def add_wear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tool wear related features।"""
    df = df.copy()
    # Wear rate normalized by speed
    df["wear_speed_ratio"] = df["tool_wear"] / (df["rotational_speed"] + 1e-8)
    # Torque per wear unit — high value means stress
    df["torque_per_wear"] = df["torque"] / (df["tool_wear"] + 1)
    return df


def run_feature_engineering(config_path: str = "configs/data_config.yaml") -> None:
    """Feature engineering pipeline চালাও।"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]

    for split in ["processed_train", "processed_test"]:
        path = data_cfg[split]
        logger.info(f"Feature engineering: {path}")

        df = pd.read_parquet(path)
        original_cols = df.shape[1]

        # নতুন features add করো
        df = add_temperature_features(df)
        df = add_power_features(df)
        df = add_wear_features(df)

        new_cols = df.shape[1] - original_cols
        logger.info(f"Added {new_cols} new features. Total: {df.shape[1]} columns")

        # Save করো (overwrite processed file)
        df.to_parquet(path, index=False)
        logger.info(f"✅ Saved: {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_feature_engineering()