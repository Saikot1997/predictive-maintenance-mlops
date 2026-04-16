"""
Model training with full MLflow experiment tracking।
RandomForest এবং XGBoost দুটো model train হবে।
Best model automatically MLflow Model Registry তে register হবে।
"""
import logging
import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    average_precision_score
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_configs() -> tuple:
    with open("configs/data_config.yaml") as f:
        data_cfg = yaml.safe_load(f)
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/training_config.yaml") as f:
        train_cfg = yaml.safe_load(f)
    return data_cfg, model_cfg, train_cfg


def load_data(data_cfg: dict) -> tuple:
    """Processed data load করো।"""
    train = pd.read_parquet(data_cfg["data"]["processed_train"])
    test = pd.read_parquet(data_cfg["data"]["processed_test"])

    target = "target"
    feature_cols = [c for c in train.columns if c != target]

    X_train = train[feature_cols]
    y_train = train[target]
    X_test = test[feature_cols]
    y_test = test[target]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Failure rate (train): {y_train.mean():.3f}")
    return X_train, X_test, y_train, y_test, feature_cols


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """সব evaluation metrics calculate করো।"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str) -> str:
    """Confusion matrix plot তৈরি করো।"""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Fail", "Fail"])
    ax.set_yticklabels(["No Fail", "Fail"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    path = f"/tmp/cm_{model_name}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def train_random_forest(X_train, y_train, X_test, y_test,
                         model_cfg: dict, train_cfg: dict) -> tuple:
    """RandomForest train করো এবং MLflow এ log করো।"""
    rf_params = model_cfg["models"]["random_forest"]
    exp_name = train_cfg["training"]["experiment_name"]

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="random_forest") as run:
        logger.info("Training RandomForest...")

        # Sample weights (imbalanced data handle)
        sample_w = compute_sample_weight("balanced", y_train)

        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train, sample_weight=sample_w)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)

        # MLflow এ log করো
        mlflow.log_params(rf_params)
        mlflow.log_metrics(metrics)

        # Confusion matrix artifact
        cm_path = plot_confusion_matrix(y_test, y_pred, "RandomForest")
        mlflow.log_artifact(cm_path, "plots")

        # Feature importance log করো
        fi = pd.Series(model.feature_importances_,
                       index=X_train.columns).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.head(10).plot(kind="bar", ax=ax, color="#4f86c6")
        ax.set_title("Top 10 Feature Importances")
        ax.set_xlabel("Feature"); ax.set_ylabel("Importance")
        plt.tight_layout()
        fi_path = "/tmp/feature_importance_rf.png"
        plt.savefig(fi_path, dpi=100)
        plt.close()
        mlflow.log_artifact(fi_path, "plots")

        # Model save করো
        signature = infer_signature(X_train, y_pred)
        mlflow.sklearn.log_model(
            model, "model",
            signature=signature,
            registered_model_name=None,  # Best model পরে register হবে
        )

        run_id = run.info.run_id
        logger.info(f"RandomForest | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return model, metrics, run_id


def train_xgboost(X_train, y_train, X_test, y_test,
                  model_cfg: dict, train_cfg: dict) -> tuple:
    """XGBoost train করো এবং MLflow এ log করো।"""
    xgb_params = model_cfg["models"]["xgboost"]
    exp_name = train_cfg["training"]["experiment_name"]

    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="xgboost") as run:
        logger.info("Training XGBoost...")

        model = XGBClassifier(
            **xgb_params,
            verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_prob)

        mlflow.log_params(xgb_params)
        mlflow.log_metrics(metrics)

        cm_path = plot_confusion_matrix(y_test, y_pred, "XGBoost")
        mlflow.log_artifact(cm_path, "plots")

        signature = infer_signature(X_train, y_pred)
        mlflow.xgboost.log_model(
            model, "model",
            signature=signature,
            registered_model_name=None,
        )

        run_id = run.info.run_id
        logger.info(f"XGBoost | F1: {metrics['f1_score']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f}")

    return model, metrics, run_id


def register_best_model(rf_metrics, xgb_metrics, rf_run_id, xgb_run_id,
                        model_cfg: dict) -> None:
    """Best model MLflow Model Registry তে register করো।"""
    metric = model_cfg["models"]["best_model_metric"]
    model_name = model_cfg["models"]["registered_model_name"]

    if rf_metrics[metric] >= xgb_metrics[metric]:
        best_run_id = rf_run_id
        best_model = "RandomForest"
        best_score = rf_metrics[metric]
    else:
        best_run_id = xgb_run_id
        best_model = "XGBoost"
        best_score = xgb_metrics[metric]

    logger.info(f"🏆 Best Model: {best_model} | {metric}: {best_score:.4f}")

    # Model register করো
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, model_name)
    logger.info(f"✅ Model registered: {model_name} v{registered.version}")

    # MLflow 2.x alias API ব্যবহার করো (deprecated stage API এর পরিবর্তে)
    # transition_model_version_stage() MLflow 2.38+ এ remove হয়েছে
    client = mlflow.MlflowClient()
    try:
        # Modern approach: alias set করো ("champion" = production equivalent)
        client.set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=registered.version,
        )
        logger.info(f"✅ Model alias 'champion' set → version {registered.version}")
    except Exception:
        # Fallback for older MLflow versions (< 2.x)
        client.transition_model_version_stage(
            name=model_name,
            version=registered.version,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info(f"✅ Model promoted to Production stage (legacy stage API)!")


def main():
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )

    data_cfg, model_cfg, train_cfg = load_configs()
    X_train, X_test, y_train, y_test, feature_cols = load_data(data_cfg)

    # Train করো
    rf_model, rf_metrics, rf_run_id = train_random_forest(
        X_train, y_train, X_test, y_test, model_cfg, train_cfg
    )
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost(
        X_train, y_train, X_test, y_test, model_cfg, train_cfg
    )

    # Best model register করো
    register_best_model(rf_metrics, xgb_metrics, rf_run_id, xgb_run_id, model_cfg)

    # Local backup save করো
    save_path = model_cfg["models"]["model_save_path"]
    Path(save_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, f"{save_path}/random_forest.pkl")
    joblib.dump(xgb_model, f"{save_path}/xgboost.pkl")
    logger.info(f"Models saved to: {save_path}")


if __name__ == "__main__":
    main()