"""
BentoML 1.3.x service — ML-native model serving.
Compatible with BentoML 1.3.22+.
"""

from typing import Literal

import bentoml
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


def save_model_to_bentoml() -> str:
    """Save trained model to BentoML model store."""
    import joblib

    # Load from local pkl (most reliable — avoids MLflow registry dependency)
    model_path = "models/trained/xgboost.pkl"
    sklearn_model = joblib.load(model_path)
    bento_model = bentoml.sklearn.save_model(
        "pm_model",
        sklearn_model,
        metadata={
            "description": "Predictive Maintenance Model",
            "dataset": "AI4I 2020",
        },
    )
    print(f"Saved to BentoML: {bento_model.tag}")
    return str(bento_model.tag)


class SensorInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float
    type: Literal["L", "M", "H"]


class PredictionOutput(BaseModel):
    prediction: int
    probability: float
    risk_level: str


def _build_features(inp: SensorInput) -> pd.DataFrame:
    """Build feature vector matching training pipeline."""
    try:
        encoder = joblib.load("models/trained/label_encoder.pkl")
        type_enc = int(encoder.transform([inp.type])[0])
    except Exception:
        type_enc = {"H": 0, "L": 1, "M": 2}.get(inp.type, 1)

    power = inp.torque * inp.rotational_speed * (2 * np.pi / 60)
    df = pd.DataFrame(
        [
            {
                "Type": type_enc,
                "air_temperature": inp.air_temperature,
                "process_temperature": inp.process_temperature,
                "rotational_speed": inp.rotational_speed,
                "torque": inp.torque,
                "tool_wear": inp.tool_wear,
                "temp_diff": inp.process_temperature - inp.air_temperature,
                "temp_ratio": inp.process_temperature / (inp.air_temperature + 1e-8),
                "power_consumption": power,
                "power_wear_interaction": power * inp.tool_wear,
                "wear_speed_ratio": inp.tool_wear / (inp.rotational_speed + 1e-8),
                "torque_per_wear": inp.torque / (inp.tool_wear + 1),
            }
        ]
    )
    try:
        scaler = joblib.load("models/trained/scaler.pkl")
        numeric_cols = [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ]
        df[numeric_cols] = scaler.transform(df[numeric_cols])
    except Exception:
        pass
    return df


@bentoml.service(name="predictive_maintenance_svc")
class PredictiveMaintenanceService:
    """BentoML 1.3.x service for predictive maintenance."""

    def __init__(self):
        pm_model_ref = bentoml.models.get("pm_model:latest")
        self.model = pm_model_ref.load_model()

    @bentoml.api
    def predict(self, input_data: SensorInput) -> PredictionOutput:
        """Predict machine failure from sensor readings."""
        features = _build_features(input_data)
        prediction = int(self.model.predict(features)[0])
        try:
            probability = float(self.model.predict_proba(features)[0][1])
        except AttributeError:
            probability = float(prediction)
        risk = "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.3 else "LOW"
        return PredictionOutput(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk,
        )
