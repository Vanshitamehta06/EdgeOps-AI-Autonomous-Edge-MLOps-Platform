"""
monitor.py

Monitors live edge inference logs for statistical data drift using Evidently AI.
Compares incoming API telemetry against the baseline training dataset to detect
when the active model is operating in an unseen environment.
"""

import json
import logging
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from evidently import Report
from evidently.presets import DataDriftPreset

# Local imports
import config
import database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback thresholds if not explicitly defined in config.py
DRIFT_THRESHOLD = getattr(config, 'DRIFT_THRESHOLD', 0.3)  # 30% of features drifted
MIN_LOG_SAMPLES = getattr(config, 'MIN_LOG_SAMPLES', 50)   # Minimum samples needed for statistical tests

def get_active_model_version() -> Optional[int]:
    """Retrieves the currently deployed model version from the SQLite registry."""
    deployment = database.get_current_deployment()
    if not deployment:
        logger.warning("No active deployment found. Cannot monitor drift.")
        return None
    return deployment["version_number"]

def load_reference_data() -> pd.DataFrame:
    """
    Loads the original dataset to act as the baseline distribution.
    Strictly filters out dropped columns and targets to match the API payload exactly.
    """
    logger.info(f"Loading reference dataset from {config.DATASET_PATH}")
    df = pd.read_csv(config.DATASET_PATH)
    
    # Drop irrelevant columns (UDI, Product ID, etc.)
    cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Drop the target column; we are measuring *Data* Drift, not Concept Drift here
    if config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])
        
    return df

def load_current_telemetry(active_version: int) -> Optional[pd.DataFrame]:
    """
    Parses the JSONL inference logs and extracts the payloads processed by the active model.
    """
    log_file = config.LOGS_DIR / "inference_logs.jsonl"
    
    if not log_file.exists():
        logger.info("No inference logs found yet.")
        return None

    # Read JSON Lines file
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
                
    if not logs:
        return None

    # Convert to DataFrame and filter by the active model version
    df_logs = pd.DataFrame(logs)
    df_current_model = df_logs[df_logs["model_version"] == active_version].copy()
    
    if len(df_current_model) < MIN_LOG_SAMPLES:
        logger.info(
            f"Not enough telemetry data for statistical drift detection. "
            f"Have {len(df_current_model)}, need {MIN_LOG_SAMPLES}."
        )
        return None
        
    # Keep only the feature columns expected by the model (ignore timestamp, prediction, etc.)
    # Note: The API explicitly logs features using exact dataset names (e.g., 'Air temperature [K]')
    reference_columns = load_reference_data().columns
    feature_logs = df_current_model[[col for col in reference_columns if col in df_current_model.columns]]
    
    return feature_logs

def analyze_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes Evidently AI's DataDriftPreset to compare distributions.
    Strictly runs on CPU and returns a parsed dictionary of the report.
    """
    logger.info("Executing Evidently AI Data Drift analysis...")
    
    # Initialize the Evidently report
    drift_report = Report(metrics=[DataDriftPreset()])
    
    # Run the report
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Extract results as dictionary to avoid heavy HTML rendering
    report_dict = drift_report.as_dict()
    
    # Navigate the nested Evidently dictionary structure to find the drift flag
    # Metrics [0] is the DataDriftTable metric
    drift_metrics = report_dict["metrics"][0]["result"]
    
    share_of_drifted_columns = drift_metrics["share_of_drifted_columns"]
    dataset_drift = drift_metrics["dataset_drift"]
    
    logger.info(f"Drift Analysis Complete. Drifted features: {share_of_drifted_columns*100:.1f}%.")
    
    return {
        "drift_detected": dataset_drift,
        "drift_share": share_of_drifted_columns,
        "drifted_features": drift_metrics["number_of_drifted_columns"]
    }

def run_monitor() -> bool:
    """
    Orchestrates the monitoring pipeline.
    Returns True if critical drift is detected, signaling the optimizer to run.
    """
    logger.info("=== Starting EdgeOps Monitoring Cycle ===")
    
    active_version = get_active_model_version()
    if not active_version:
        return False
        
    current_data = load_current_telemetry(active_version)
    if current_data is None:
        return False
        
    reference_data = load_reference_data()
    
    # Ensure columns match strictly
    common_cols = list(set(reference_data.columns).intersection(set(current_data.columns)))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]
    
    # Perform Analysis
    drift_results = analyze_data_drift(reference_data, current_data)
    
    drift_state = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_detected": bool(drift_results["drift_detected"]),
        "drift_share": float(drift_results["drift_share"])
    }
    with open(config.BASE_DIR / "drift_state.json", "w") as f:
        json.dump(drift_state, f)
        
    if drift_results["drift_detected"] or drift_results["drift_share"] >= DRIFT_THRESHOLD:
        logger.warning(
            f"CRITICAL: Data drift detected! "
            f"{drift_results['drift_share']*100:.1f}% of features have shifted."
        )
        return True
    else:
        logger.info("System healthy. No significant data drift detected.")
        return False

if __name__ == "__main__":
    # Can be run manually or as a cron job
    run_monitor()