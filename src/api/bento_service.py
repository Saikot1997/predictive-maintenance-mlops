"""
BentoML 1.x service — ML-native model serving।
BentoML 1.x এর নতুন @bentoml.service decorator API ব্যবহার করা হয়েছে।
MLflow registry থেকে model নিয়ে BentoML এ save করে serve করে।
"""

import os
from typing import Literal

import bentoml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


# ── Step 1: MLflow থেকে model নিয়ে BentoML এ save করো ──
def save_model_to_bentoml() -> str:
    """MLflow Production model BentoML model store এ save করো।
    এটা একবারই run করতে হবে।
    Usage: python -c "from src.api.bento_service import save_model_to_bentoml; save_model_to_bentoml
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_name = os.getenv("MODEL_NAME", "predictive_maintenance_model")
    # Try alias first (MLflow 2.x), fallback to legacy stage
    try:
        sklearn_model = mlflow.sklearn.load_model(f"models:/{model_name}@champion")
    except Exception:
        sklearn_model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
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


# ── Step 2: Input/Output Schema ──────────────────────────
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


# ── Step 3: BentoML 1.x Service (new decorator API) ──────
# BentoML 1.x তে class-based service ব্যবহার করতে হয়।
# পুরনো bentoml.Service() এবং bentoml.io — 1.x তে নেই।
@bentoml.service(
    name="predictive_maintenance_svc",
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class PredictiveMaintenanceService:
    """BentoML 1.x class-based service।"""

    def __init__(self):
        # Load model lazily in __init__, not at class definition time.
        # bentoml.models.get() at class scope crashes on import if the model
        # store is empty (before save_model_to_bentoml() has been called).
        try:
            pm_model_ref = bentoml.models.get("pm_model:latest")
            self.model = pm_model_ref.load_model()
        except Exception as e:
            raise RuntimeError(
                f"BentoML model 'pm_model:latest' not found in store. "
                f"Run save_model_to_bentoml() first. Original error: {e}"
            )

    def _build_features(self, inp: SensorInput) -> "pd.DataFrame":
        """Feature vector তৈরি করো।
        DataFrame use করা হয়েছে যাতে column names match হয় training এর সাথে।
        Raw numpy array use করলে XGBoost feature order mismatch হয়।
        IMPORTANT: LabelEncoder alphabetically sorts: H=0, L=1, M=2.
        Hardcoding L=0,M=1,H=2 would be WRONG — always load the saved encoder.
        """
        import joblib
        import pandas as pd

        # Load saved encoder to get correct mapping (H=0, L=1, M=2 alphabetically)
        try:
            encoder = joblib.load("models/trained/label_encoder.pkl")
            type_enc = int(encoder.transform([inp.type])[0])
        except Exception:
            # Fallback: alphabetical order H=0, L=1, M=2
            type_enc = {"H": 0, "L": 1, "M": 2}.get(inp.type, 1)
        power = inp.torque * inp.rotational_speed * (2 * np.pi / 60)
        temp_diff = inp.process_temperature - inp.air_temperature
        temp_ratio = inp.process_temperature / (inp.air_temperature + 1e-8)
        wear_speed = inp.tool_wear / (inp.rotational_speed + 1e-8)
        torque_wear = inp.torque / (inp.tool_wear + 1)

        # Build DataFrame with feature engineering first, then scale
        df = pd.DataFrame(
            [
                {
                    "Type": type_enc,
                    "air_temperature": inp.air_temperature,
                    "process_temperature": inp.process_temperature,
                    "rotational_speed": inp.rotational_speed,
                    "torque": inp.torque,
                    "tool_wear": inp.tool_wear,
                    "temp_diff": temp_diff,
                    "temp_ratio": temp_ratio,
                    "power_consumption": power,
                    "power_wear_interaction": power * inp.tool_wear,
                    "wear_speed_ratio": wear_speed,
                    "torque_per_wear": torque_wear,
                }
            ]
        )
        # Apply saved scaler to numeric columns
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
            pass  # Fallback: serve unscaled if scaler unavailable
        return df

    @bentoml.api
    def predict(self, input_data: SensorInput) -> PredictionOutput:
        """Prediction endpoint।"""
        features = self._build_features(input_data)
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
