"""
Evidently AI দিয়ে data drift detection।
Reference data (training) vs Current data (production) compare করে।
HTML report তৈরি করে — browser এ দেখা যাবে।
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.metrics import DatasetDriftMetric
from evidently.report import Report

logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_path: str = "data/processed/train.parquet",
    current_path: str = "data/processed/test.parquet",
    output_dir: str = "data/reports/",
    config_path: str = "configs/training_config.yaml",
) -> dict:
    """Data drift report তৈরি করো।

    Returns: drift summary dict (drift detected: True/False, score)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    threshold = cfg["monitoring"]["drift_threshold"]

    # Data load করো
    reference = pd.read_parquet(reference_path)
    current = pd.read_parquet(current_path)

    # Target column আলাদা করো
    target_col = "target"
    feature_cols = [c for c in reference.columns if c != target_col]

    # ── Report 1: Data Drift ────────────────────────────
    drift_report = Report(
        metrics=[
            DataDriftPreset(num_stattest="ks", cat_stattest="chi2"),
            DatasetDriftMetric(threshold=threshold),
            DataQualityPreset(),
        ]
    )
    drift_report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    drift_html = f"{output_dir}/drift_report_{ts}.html"
    drift_report.save_html(drift_html)
    logger.info(f"Drift report saved: {drift_html}")

    # ── Report 2: Target Drift (label distribution change) ──
    if target_col in reference.columns and target_col in current.columns:
        from evidently import ColumnMapping

        col_map = ColumnMapping(target=target_col)
        target_report = Report(
            metrics=[
                TargetDriftPreset(),
            ]
        )
        target_report.run(
            reference_data=reference[[target_col]],
            current_data=current[[target_col]],
            column_mapping=col_map,
        )
        target_html = f"{output_dir}/target_drift_{ts}.html"
        target_report.save_html(target_html)
        logger.info(f"Target drift report saved: {target_html}")

    # Summary extract করো
    # IMPORTANT: as_dict()["metrics"] is a flat list — DataDriftPreset expands to
    # multiple entries (one per column + aggregates). We must search by metric class
    # name, NOT use a hardcoded index like [1].
    drift_result = drift_report.as_dict()
    dataset_drift_result = None
    for metric_entry in drift_result["metrics"]:
        metric_id = metric_entry.get("metric", "")
        if "DatasetDriftMetric" in metric_id:
            dataset_drift_result = metric_entry["result"]
            break
    if dataset_drift_result is None:
        logger.warning("DatasetDriftMetric result not found in report — using defaults")
        dataset_drift_result = {}

    summary = {
        "drift_detected": dataset_drift_result.get("dataset_drift", False),
        "drift_score": dataset_drift_result.get("share_drifted_columns", 0.0),
        "drifted_columns": dataset_drift_result.get("number_of_drifted_columns", 0),
        "total_columns": dataset_drift_result.get("number_of_columns", 0),
        "report_path": drift_html,
        "threshold": threshold,
    }

    if summary["drift_detected"]:
        logger.warning(
            f"⚠️  DATA DRIFT DETECTED! "
            f"Score: {summary['drift_score']:.3f} > threshold: {threshold}"
        )
        logger.warning("Consider retraining the model!")
    else:
        logger.info(f"✅ No significant drift. Score: {summary['drift_score']:.3f}")

    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    summary = generate_drift_report()
    print(f"\nDrift Summary: {summary}")
