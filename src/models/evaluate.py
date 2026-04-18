"""
Trained model এর comprehensive evaluation।
সব metrics calculate করো এবং report তৈরি করো।
"""

import json
import logging
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import classification_report

load_dotenv()
logger = logging.getLogger(__name__)


def load_production_model(model_name: str):
    """MLflow registry থেকে champion/Production model load করো।
    MLflow 2.x alias ("champion") আগে try করো, না থাকলে legacy stage ("Production") ব্যবহার করো।
    """
    client = mlflow.MlflowClient()
    # Try alias first (MLflow 2.x modern approach)
    try:
        model_uri = f"models:/{model_name}@champion"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded: {model_name}@champion")
        return model
    except Exception:
        pass
    # Fallback to legacy stage API
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise ValueError(
            f"No model found for '{model_name}' (tried alias 'champion' and stage 'Production')"
        )
    model_uri = f"models:/{model_name}/Production"
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info(f"Loaded: {model_name} v{versions[0].version} (stage=Production)")
    return model


def evaluate_model(
    model_name: str = "predictive_maintenance_model",
    config_path: str = "configs/data_config.yaml",
) -> dict:
    """Production model evaluate করো এবং metrics save করো।"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    with open(config_path) as f:
        data_cfg = yaml.safe_load(f)["data"]

    # Test data load
    test = pd.read_parquet(data_cfg["processed_test"])
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    # Model load
    model = load_production_model(model_name)

    # Predict
    y_pred = model.predict(X_test)

    # Report
    report = classification_report(
        y_test, y_pred, target_names=["No Failure", "Failure"], output_dict=True
    )

    # Save metrics as JSON (DVC metrics tracking এর জন্য)
    metrics = {
        "precision": report["Failure"]["precision"],
        "recall": report["Failure"]["recall"],
        "f1_score": report["Failure"]["f1-score"],
        "accuracy": report["accuracy"],
    }
    Path("data/reports").mkdir(exist_ok=True)
    with open("data/reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Evaluation Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_model()
