"""
Prediction logic — FastAPI ও BentoML দুটোতেই ব্যবহার হবে।
MLflow registry থেকে latest Production model load করে predict করে।
"""

import logging
import os

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Module level model cache — প্রতিটা request এ reload না করতে
_model = None
_scaler = None
_encoder = None


def get_model():
    """Singleton pattern: model একবারই load হবে।"""
    global _model
    if _model is None:
        model_name = os.getenv("MODEL_NAME", "predictive_maintenance_model")
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        try:
            # Try alias first (MLflow 2.x), fallback to legacy stage
            try:
                _model = mlflow.pyfunc.load_model(f"models:/{model_name}@champion")
                logger.info(f"Model loaded from MLflow registry: {model_name}@champion")
            except Exception:
                _model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
                logger.info(
                    f"Model loaded from MLflow registry: {model_name}/Production"
                )
        except Exception as e:
            # Fallback: local pkl file থেকে load করো
            logger.warning(f"MLflow unavailable: {e}. Loading local model.")
            local_path = "models/trained/xgboost.pkl"
            _model = joblib.load(local_path)
            logger.info(f"Local model loaded: {local_path}")
    return _model


def get_preprocessors():
    """Scaler ও encoder load করো।"""
    global _scaler, _encoder
    if _scaler is None:
        _scaler = joblib.load("models/trained/scaler.pkl")
    if _encoder is None:
        _encoder = joblib.load("models/trained/label_encoder.pkl")
    return _scaler, _encoder


def preprocess_input(data: dict) -> pd.DataFrame:
    """Single prediction input preprocess করো।
    Order must match training pipeline exactly:
    encode → rename → feature engineering → scale numeric
    """
    scaler, encoder = get_preprocessors()

    # Type encode করো
    type_encoded = encoder.transform([data["type"]])[0]

    # DataFrame তৈরি করো (column names must match training)
    df = pd.DataFrame(
        [
            {
                "Type": float(type_encoded),
                "air_temperature": data["air_temperature"],
                "process_temperature": data["process_temperature"],
                "rotational_speed": data["rotational_speed"],
                "torque": data["torque"],
                "tool_wear": data["tool_wear"],
            }
        ]
    )

    # Feature engineering FIRST (same order as training pipeline)
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["temp_ratio"] = df["process_temperature"] / (df["air_temperature"] + 1e-8)
    df["power_consumption"] = df["torque"] * df["rotational_speed"] * (2 * np.pi / 60)
    df["power_wear_interaction"] = df["power_consumption"] * df["tool_wear"]
    df["wear_speed_ratio"] = df["tool_wear"] / (df["rotational_speed"] + 1e-8)
    df["torque_per_wear"] = df["torque"] / (df["tool_wear"] + 1)

    # Scale numeric columns AFTER feature engineering (matches training order)
    numeric_cols = [
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


FAILURE_TYPES = {
    0: "No Failure",
    1: "Machine Failure",
    # Note: this is a binary classifier — prediction is 0 or 1 only.
    # Specific failure type classification (TWF/HDF/PWF/OSF/RNF)
    # would require a separate multi-label model.
}


def predict(data: dict) -> dict:
    """Single sample prediction।
    Returns: prediction (0/1), probability, failure_type string
    """
    model = get_model()
    df = preprocess_input(data)

    prediction = int(model.predict(df)[0])

    # Probability
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(df)[0][1])
    else:
        # MLflow pyfunc এর জন্য
        try:
            prob = float(model.predict(df)[0])
        except Exception:
            prob = float(prediction)

    failure_type = FAILURE_TYPES.get(prediction, "Unknown")

    return {
        "prediction": prediction,
        "probability": round(prob, 4),
        "failure_type": failure_type,
        "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW",
    }
