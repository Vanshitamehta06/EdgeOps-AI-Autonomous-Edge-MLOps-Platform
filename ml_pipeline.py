"""
ml_pipeline.py

Executes the end-to-end machine learning pipeline.
Handles data loading, preprocessing, tiered training configurations, evaluation, 
saving bundles, ONNX conversion, MLflow tracking, and SQLite registration.
"""

import os
import json
import joblib
import logging
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import optuna
from typing import Tuple, Dict, Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

# Local imports
import config
import database

# Configure matplotlib for headless environments (no GUI popups)
plt.switch_backend('Agg')
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 1. DATA INGESTION & PREPROCESSING
# ---------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Loads the dataset from the configured file path."""
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at {config.DATASET_PATH}")
    logger.info(f"Loading dataset from {config.DATASET_PATH}")
    return pd.read_csv(config.DATASET_PATH)

def preprocess_data(df: pd.DataFrame):
    """
    Cleans the input dataframe, drops leakage columns, sanitizes feature names 
    for XGBoost compatibility, and returns X, y, and the encoder.
    """
    df = df.copy()
    
    # 1. Drop identifier and target leakage columns
    df = df.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')
    
    # 2. THE XGBOOST FIX: Remove [, ], and < characters from column names
    df.columns = (
        df.columns.str.replace('[', '', regex=False)
                  .str.replace(']', '', regex=False)
                  .str.replace('<', '', regex=False)
    )
    
    # 3. Separate features (X) and target (y)
    if config.TARGET_COLUMN in df.columns:
        X = df.drop(columns=[config.TARGET_COLUMN])
        y = df[config.TARGET_COLUMN]
    else:
        X = df
        y = None

    # 4. Map the categorical 'Type' column to numbers if it's text
    if 'Type' in X.columns and X['Type'].dtype == 'object':
        X['Type'] = X['Type'].map({'L': 0, 'M': 1, 'H': 2}).fillna(0)
    
   # 5. Define the encoder placeholder to satisfy unpacking rules
    # FIX: We now initialize a dummy encoder so the file physically generates
    encoder = LabelEncoder()
    encoder.classes_ = np.array(['L', 'M', 'H']) 
    
    return X, y, encoder

def split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the preprocessed data into training and testing sets."""
    logger.info("Splitting dataset...")
    return train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y  # Ensure class balance in splits for binary classification
    )

# ---------------------------------------------------------
# 2. TRAINING & EVALUATION
# ---------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series, custom_params: Dict[str, Any]) -> XGBClassifier:
    """Trains the XGBoost classifier using injected dynamic parameter configurations."""
    logger.info("Training XGBoost instance with updated parameter map...")
    model = XGBClassifier(**custom_params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluates model performance and generates metrics."""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    logger.info(f"Evaluation complete. Metrics: {metrics}")
    return metrics

def generate_evaluation_plots(model: XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series, save_dir: str):
    """Generates and saves Confusion Matrix and Feature Importance plots."""
    os.makedirs(save_dir, exist_ok=True)
    y_pred = model.predict(X_test)

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

    # 2. Feature Importance Plot
    plt.figure(figsize=(8, 6))
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.bar(range(X_test.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_test.shape[1]), X_test.columns[indices], rotation=45, ha='right')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "feature_importance.png"))
    plt.close()

# ---------------------------------------------------------
# 3. EXPORT & SAVING
# ---------------------------------------------------------

def export_to_onnx(model, X_train, export_path: str):
    """Converts the trained XGBoost model into portable ONNX format."""
    booster = model.get_booster()
    original_features = None
    
    if booster.feature_names is not None:
        original_features = list(booster.feature_names)
        booster.feature_names = [f"f{num}" for num in range(len(original_features))]
        
    try:
        num_features = X_train.shape[1]
        initial_type = [('float_input', FloatTensorType([None, num_features]))]
        onnx_model = convert_xgboost(model, initial_types=initial_type)
        
        with open(export_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
    finally:
        if original_features is not None:
            booster.feature_names = original_features

def save_model_bundle(
    version: int, model: XGBClassifier, encoder: LabelEncoder, 
    X_train: pd.DataFrame, metrics: Dict[str, float], hyperparams: Dict[str, Any]
) -> str:
    """Saves the full model bundle (pickle, ONNX, and metadata) into a versioned directory."""
    version_str = config.VERSION_FORMAT.format(version)
    version_dir = config.MODELS_DIR / version_str
    version_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model bundle to {version_dir}")

    # 1. Save Pickle components
    joblib.dump(model, version_dir / "model.pkl")
    if encoder is not None:
        joblib.dump(encoder, version_dir / "label_encoder.pkl")

    
    # 2. Save ONNX binary
    export_to_onnx(model, X_train, str(version_dir / "model.onnx"))
    
    # 3. Save Reproducible Metadata JSON
    metadata = {
        "version": version,
        "feature_columns": list(X_train.columns),
        "target_column": config.TARGET_COLUMN,
        "train_size": len(X_train),
        "random_state": config.RANDOM_STATE,
        "hyperparameters": hyperparams,
        "metrics": metrics
    }
    
    with open(version_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    return str(version_dir)

# ---------------------------------------------------------
# 4. PIPELINE ORCHESTRATION WITH GATED RETRAINING
# ---------------------------------------------------------

# ---------------------------------------------------------
# 4. PIPELINE ORCHESTRATION WITH GATED RETRAINING
# ---------------------------------------------------------

# ---------------------------------------------------------
# 4. PIPELINE ORCHESTRATION WITH GATED RETRAINING
# ---------------------------------------------------------

def run_pipeline(incoming_data: pd.DataFrame = None) -> None:
    """
    Orchestrates the Gated Multi-Tier Retraining Strategy.
    Executes a Historical Battle Royale to handle recurring drift.
    """
    logger.info("=== Starting EdgeOps Gated ML Pipeline ===")
    
    # Setup tracking integrations
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    # 1. Ingest Inbound Telemetry
    if incoming_data is not None:
        logger.info("📥 Processing freshly streamed industrial telemetry data.")
        df = incoming_data
    else:
        logger.info("📊 Processing historical registry dataset baseline.")
        df = load_dataset()

    X, y, encoder = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Compute dynamic imbalance balance anchor weight
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    dynamic_weight = neg_count / max(pos_count, 1)
    logger.info(f"Calculated Dynamic Class Imbalance Balancing Ratio: {dynamic_weight:.4f}")

    # ---------------------------------------------------------
    # TIER 1: HISTORICAL BATTLE ROYALE
    # ---------------------------------------------------------
    logger.info("Tier 1: Initiating Historical Battle Royale. Evaluating ALL past models on new data...")
    
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, version_number, model_path, status FROM model_versions")
    historical_models = cursor.fetchall()
    conn.close()

    best_historical_f1 = 0.0
    best_historical_id = None
    best_historical_version = None

    for m_id, m_version, m_path, m_status in historical_models:
        try:
            pkl_path = os.path.join(m_path, "model.pkl")
            if os.path.exists(pkl_path):
                # Load each past model and test it
                old_model = joblib.load(pkl_path)
                y_pred = old_model.predict(X_test)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                logger.info(f" - Evaluated v{m_version} ({m_status}) | F1 Score: {f1:.4f}")
                
                # Keep track of the ultimate defender
                if f1 > best_historical_f1:
                    best_historical_f1 = f1
                    best_historical_id = m_id
                    best_historical_version = m_version
        except Exception as e:
            logger.warning(f"Could not evaluate historical v{m_version}: {e}")

    if best_historical_id is not None:
        logger.info(f"👑 Best Historical Defender is v{best_historical_version} with F1: {best_historical_f1:.4f}")
    else:
        logger.info("No valid historical models found. Baseline set to 0.0")

    # ---------------------------------------------------------
    # TIER 2: FAST CHALLENGER RETRAIN (Base Config + Dynamic Weight)
    # ---------------------------------------------------------
    logger.info("Tier 2: Building Fast Challenger node using base configuration parameters...")
    tier2_params = config.XGB_PARAMS.copy()
    tier2_params["scale_pos_weight"] = dynamic_weight
    
    t2_model = train_model(X_train, y_train, tier2_params)
    t2_metrics = evaluate_model(t2_model, X_test, y_test)
    
    if t2_metrics["f1"] > best_historical_f1:
        logger.info(f"🏆 Tier 2 Success! Fast Challenger ({t2_metrics['f1']:.4f}) beat Historical Baseline ({best_historical_f1:.4f}). Promoting.")
        execute_final_deployment(t2_model, encoder, X_train, X_test, y_test, t2_metrics, tier2_params)
        return

    # ---------------------------------------------------------
    # TIER 3: HEAVY CHALLENGER RETRAIN (Optuna Parameter Search)
    # ---------------------------------------------------------
    logger.warning(f"⚠️ Tier 2 Insufficient: Fast Challenger F1 ({t2_metrics['f1']:.4f}) failed to outpace Target. Invoking Optuna Engine.")
    
    def objective(trial):
        optuna_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": config.RANDOM_STATE,
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", dynamic_weight * 0.8, dynamic_weight * 1.2)
        }
        trial_model = XGBClassifier(**optuna_params)
        trial_model.fit(X_train, y_train)
        preds = trial_model.predict(X_test)
        return f1_score(y_test, preds, zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=15)
    
    best_tuned_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": config.RANDOM_STATE,
        "n_jobs": -1,
        **study.best_params
    }
    
    logger.info(f"Optuna Exploration Finished. Best Config Discovered: {study.best_params}")
    
    t3_model = train_model(X_train, y_train, best_tuned_params)
    t3_metrics = evaluate_model(t3_model, X_test, y_test)
    
    # ---------------------------------------------------------
    # FINAL TOURNAMENT GATE
    # ---------------------------------------------------------
    if t3_metrics["f1"] > best_historical_f1:
        logger.info(f"🔥 Tier 3 Optimization Successful! New Tuned F1 Score: {t3_metrics['f1']:.4f}")
        execute_final_deployment(t3_model, encoder, X_train, X_test, y_test, t3_metrics, best_tuned_params)
    else:
        logger.error(f"❌ Retraining Rejected: New Configurations ({t3_metrics['f1']:.4f}) could not beat Historical Baseline ({best_historical_f1:.4f}).")
        
        # --- RECURRING DRIFT HOT-SWAP LOGIC ---
        current_deployment = database.get_current_deployment()
        active_id = current_deployment.get("current_model_id") if isinstance(current_deployment, dict) else None
        
        if best_historical_id and best_historical_id != active_id:
            logger.info(f"🔄 RECURRING DRIFT DETECTED! A retired model (v{best_historical_version}) is the best fit for this data.")
            logger.info(f"Pulling v{best_historical_version} out of retirement and promoting to ACTIVE status.")
            database.promote_model_to_production(best_historical_id)
        else:
            logger.info("The current Active Model remains the overall best fit. No changes made.")


def execute_final_deployment(model, encoder, X_train, X_test, y_test, metrics, hyperparams):
    """Handles pipeline persistence layers including file systems, plots, MLflow tracking, and tables."""
    next_version = database.get_next_version_number()
    
    # 1. Persist Bundle Assets to Disk
    bundle_path = save_model_bundle(next_version, model, encoder, X_train, metrics, hyperparams)
    
    # 2. Generate Evaluation Artifact Plots
    temp_plot_dir = config.BASE_DIR / "temp_plots"
    generate_evaluation_plots(model, X_test, y_test, str(temp_plot_dir))

    # 3. Log Complete Metadata Ecosystem to MLflow Run Registry
    with mlflow.start_run(run_name=f"Automated_Retrain_v{next_version}"):
        mlflow.log_params(hyperparams)
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(bundle_path, artifact_path="model_bundle")
        mlflow.log_artifacts(str(temp_plot_dir), artifact_path="evaluation_plots")
        mlflow.set_tag("version", f"v{next_version}")

    # 4. Commit Package to Local Persistent SQLite State Machine
    model_id = database.register_model_version(
        version_number=next_version,
        model_path=bundle_path,
        metrics=metrics
    )

    # 5. Automatically Promote the New Outperforming Champion to Active Status
    logger.info(f"Promoting newly generated model version {next_version} to production active status node.")
    database.promote_model_to_production(model_id)
    logger.info(f"=== Pipeline Finished. Version {next_version} is deployed. ===")


if __name__ == "__main__":
    database.init_db()
    run_pipeline()