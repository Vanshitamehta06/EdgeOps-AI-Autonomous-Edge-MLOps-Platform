"""
monitor.py

Monitors live edge inference logs for statistical data drift.
Uses Two-Sample Kolmogorov-Smirnov (KS) tests, Population Stability Index (PSI),
and Chi-Square goodness-of-fit to compare incoming live telemetry against the baseline.
Strictly CPU-optimized (<5ms calculation) for industrial edge devices.
"""

import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.stats import ks_2samp, chisquare

# Local imports
import config
import database

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - [%(name)s] - %(message)s')
logger = logging.getLogger("edgeops.monitor")

# Config thresholds
DRIFT_THRESHOLD = getattr(config, 'DRIFT_THRESHOLD', 0.3)  # 30% of features drifted
MIN_LOG_SAMPLES = getattr(config, 'MIN_LOG_SAMPLES', 50)   # Minimum samples needed for statistical tests
KS_ALPHA = 0.05                                            # Significance level for KS test
PSI_THRESHOLD = 0.25                                       # Standard industry PSI drift boundary

def get_active_model_version() -> Optional[int]:
    """Retrieves the currently deployed model version from the SQLite registry."""
    deployment = database.get_current_deployment()
    if not deployment:
        logger.warning("No active deployment found in registry. Cannot monitor drift.")
        return None
    return deployment["version_number"]

def load_reference_data() -> pd.DataFrame:
    """
    Loads the original dataset to act as the baseline distribution.
    Strictly filters out dropped columns and targets to match the API payload exactly.
    """
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Reference dataset not found at {config.DATASET_PATH}")
    df = pd.read_csv(config.DATASET_PATH)
    
    # Drop identifier columns and leakage flags
    cols_to_drop = [col for col in config.COLUMNS_TO_DROP if col in df.columns]
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Drop target column
    if config.TARGET_COLUMN in df.columns:
        df = df.drop(columns=[config.TARGET_COLUMN])
        
    return df

def load_current_telemetry(active_version: int) -> Optional[pd.DataFrame]:
    """
    Parses JSONL inference logs and extracts telemetry payloads processed by the active model.
    """
    log_file = config.LOGS_DIR / "inference_logs.jsonl"
    
    if not log_file.exists():
        logger.info("No inference logs found yet.")
        return None

    logs = []
    with open(log_file, "r") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                
    if not logs:
        return None

    df_logs = pd.DataFrame(logs)
    if "model_version" not in df_logs.columns:
        return None
        
    df_current_model = df_logs[df_logs["model_version"] == active_version].copy()
    
    if len(df_current_model) < MIN_LOG_SAMPLES:
        logger.info(
            f"Not enough telemetry data for statistical drift detection. "
            f"Have {len(df_current_model)}, need {MIN_LOG_SAMPLES}."
        )
        return None
        
    reference_columns = load_reference_data().columns
    feature_logs = df_current_model[[col for col in reference_columns if col in df_current_model.columns]]
    return feature_logs

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
    """
    Computes Population Stability Index (PSI) between reference and current distributions.
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    try:
        # Determine quantile breakpoints from reference distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Smooth with epsilon to avoid division by zero or log(0)
        eps = 1e-4
        expected_pct = (expected_counts + eps) / (len(expected) + eps * num_buckets)
        actual_pct = (actual_counts + eps) / (len(actual) + eps * num_buckets)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)
    except Exception:
        return 0.0

def analyze_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Executes statistical drift tests on each feature:
    - Numerical: 2-Sample Kolmogorov-Smirnov (KS) test (p < 0.05) + PSI (> 0.25)
    - Categorical: Chi-square test (p < 0.05)
    Returns a structured drift report matching the EdgeOps standard.
    """
    logger.info("Executing EdgeOps Statistical Data Drift Analysis...")
    drifted_features = []
    feature_details = {}

    total_columns = len(reference_data.columns)

    for col in reference_data.columns:
        ref_col = reference_data[col].dropna()
        cur_col = current_data[col].dropna()

        if len(ref_col) == 0 or len(cur_col) == 0:
            continue

        # Check if numeric
        if pd.api.types.is_numeric_dtype(ref_col):
            ref_arr = ref_col.to_numpy(dtype=float)
            cur_arr = cur_col.to_numpy(dtype=float)

            ks_stat, ks_pvalue = ks_2samp(ref_arr, cur_arr)
            psi_score = calculate_psi(ref_arr, cur_arr)

            # Drift flagged if statistically significant KS shift OR high PSI
            is_drifted = bool(ks_pvalue < KS_ALPHA or psi_score > PSI_THRESHOLD)

            feature_details[col] = {
                "test": "Kolmogorov-Smirnov + PSI",
                "ks_stat": float(ks_stat),
                "p_value": float(ks_pvalue),
                "psi": float(psi_score),
                "drift_detected": is_drifted
            }
        else:
            # Categorical distribution shift
            ref_counts = ref_col.value_counts(normalize=True)
            cur_counts = cur_col.value_counts(normalize=True)
            all_cats = list(set(ref_counts.index).union(set(cur_counts.index)))

            ref_probs = np.array([ref_counts.get(c, 1e-4) for c in all_cats])
            cur_probs = np.array([cur_counts.get(c, 1e-4) for c in all_cats])
            ref_probs /= ref_probs.sum()
            cur_probs /= cur_probs.sum()

            stat, p_val = chisquare(f_obs=cur_probs * len(cur_col), f_exp=ref_probs * len(cur_col))
            is_drifted = bool(p_val < KS_ALPHA)

            feature_details[col] = {
                "test": "Chi-Square",
                "statistic": float(stat),
                "p_value": float(p_val),
                "drift_detected": is_drifted
            }

        if is_drifted:
            drifted_features.append(col)

    share_of_drifted_columns = float(len(drifted_features) / max(total_columns, 1))
    dataset_drift = bool(share_of_drifted_columns >= DRIFT_THRESHOLD)

    logger.info(
        f"Drift Analysis Complete. Drifted features: {len(drifted_features)}/{total_columns} "
        f"({share_of_drifted_columns*100:.1f}%). Trigger condition: {dataset_drift}."
    )

    return {
        "drift_detected": dataset_drift,
        "drift_share": share_of_drifted_columns,
        "number_of_drifted_columns": len(drifted_features),
        "drifted_features": drifted_features,
        "metrics_per_feature": feature_details
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
    common_cols = [c for c in reference_data.columns if c in current_data.columns]
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]
    
    # Perform Analysis
    drift_results = analyze_data_drift(reference_data, current_data)
    
    drift_state = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_detected": bool(drift_results["drift_detected"]),
        "drift_share": float(drift_results["drift_share"]),
        "drifted_features": drift_results["drifted_features"]
    }
    with open(config.BASE_DIR / "drift_state.json", "w") as f:
        json.dump(drift_state, f, indent=4)
        
    if drift_results["drift_detected"]:
        logger.warning(
            f"CRITICAL: Data drift detected! "
            f"{drift_results['drift_share']*100:.1f}% of features have shifted."
        )
        return True
    else:
        logger.info("System healthy. No significant data drift detected.")
        return False

if __name__ == "__main__":
    run_monitor()
