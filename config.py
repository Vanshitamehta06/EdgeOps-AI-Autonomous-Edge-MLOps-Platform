"""
config.py

Centralized configuration for the EdgeOps AI MLOps platform.
Updated to support a flat root directory structure.
"""

from pathlib import Path

# ---------------------------------------------------------
# DIRECTORY & PATH MANAGEMENT
# ---------------------------------------------------------
# Base directory is the main folder (AI AT EDGE)
BASE_DIR = Path(__file__).parent.absolute()

# Files located directly in the main folder based on your workspace
DATASET_PATH = BASE_DIR / "predictive_maintenance.csv"
SQLITE_DB_PATH = BASE_DIR / "edgeops.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"

# Folders for generated outputs (MLflow, ONNX models, JSONL logs)
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
STREAM_DIR = BASE_DIR / "stream_data" # live data
# Automatically instantiate required output directories on startup
for directory in [MODELS_DIR, LOGS_DIR, STREAM_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# DATASET CONFIGURATION
# ---------------------------------------------------------
TARGET_COLUMN = "Machine failure"

# Columns to exclude from training and inference to prevent data leakage.
COLUMNS_TO_DROP = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

# ---------------------------------------------------------
# MACHINE LEARNING PARAMETERS
# ---------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Base XGBoost parameters optimized specifically for CPU execution
# config.py

# Base XGBoost parameters optimized specifically for CPU execution
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",     
    "random_state": RANDOM_STATE,
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "n_jobs": -1
    # scale_pos_weight will be injected dynamically during training!
}

# ---------------------------------------------------------
# MLOPS & SYSTEM THRESHOLDS
# ---------------------------------------------------------
# Artifact directory formatting
VERSION_FORMAT = "version_{:04d}"

# API Inference Threshold
PREDICTION_THRESHOLD = 0.50

# MLflow Tracking setup (Stores MLflow's backend in the root folder)
MLFLOW_TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow_tracking.db'}"
MLFLOW_EXPERIMENT_NAME = "EdgeOps_Predictive_Maintenance"

# Evidently AI Monitoring Thresholds
DRIFT_THRESHOLD = 0.3        
MIN_LOG_SAMPLES = 50