"""
optimizer.py

The self-optimizing engine of the EdgeOps platform.
Triggered by data drift, it automates retraining using hyperparameter tuning.
It evaluates the candidate model against the production model and automatically
promotes and hot-swaps the edge API if the candidate is superior.
"""
import os
import glob
import time 
# ... (rest of your imports)
import logging
import requests
import pandas as pd
from typing import Tuple, Dict, Any

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Local imports
import config
import database
import ml_pipeline
import monitor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Edge API endpoint for hot-swapping
EDGE_API_URL = "http://127.0.0.1:8000"

def optimize_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[XGBClassifier, Dict[str, float], Dict[str, Any]]:
    """
    Runs a CPU-optimized randomized search to find a better model configuration.
    """
    logger.info("Starting hyperparameter optimization...")
    
    # Base CPU-optimized estimator
    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",  # Strictly CPU optimized
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    # Define a lightweight search space for CPU constraints
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "n_estimators": [50, 100, 150],
        "subsample": [0.8, 1.0]
    }
    
    # Execute Randomized Search (faster than GridSearch for CPU)
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=5,          # Keep low for portfolio/simulation speed
        scoring='roc_auc', # Optimize for ROC-AUC to handle class imbalance
        cv=3,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    logger.info(f"Optimization complete. Best params: {search.best_params_}")
    
    # Evaluate the optimized model using our existing pipeline logic
    metrics = ml_pipeline.evaluate_model(best_model, X_test, y_test)
    
    return best_model, metrics, search.best_params_

def compare_and_promote(candidate_id: int, candidate_metrics: Dict[str, float]) -> bool:
    """
    Compares the newly trained candidate model against the active production model.
    Promotes the candidate if it demonstrates superior ROC-AUC.
    """
    current_deployment = database.get_current_deployment()
    
    if not current_deployment:
        logger.info("No active deployment found. Promoting candidate by default.")
        database.promote_model_to_production(candidate_id)
        return True
        
    current_roc_auc = current_deployment["roc_auc"]
    candidate_roc_auc = candidate_metrics.get("roc_auc", 0.0)
    
    logger.info(
        f"Comparing models - Current ROC-AUC: {current_roc_auc:.4f} | "
        f"Candidate ROC-AUC: {candidate_roc_auc:.4f}"
    )
    
    # We use a strict > comparison. We don't change models unless it is explicitly better.
    if candidate_roc_auc > current_roc_auc:
        logger.info("Candidate model is superior. Initiating promotion...")
        database.promote_model_to_production(candidate_id)
        return True
    else:
        logger.info("Candidate model did not outperform production. Discarding promotion.")
        return False

def trigger_edge_reload() -> None:
    """
    Fires a REST API call to the Edge Node to hot-swap the model in memory.
    """
    logger.info("Triggering Edge API reload...")
    try:
        response = requests.post(f"{EDGE_API_URL}/reload", timeout=5)
        if response.status_code == 200:
            logger.info(f"Edge Node successfully reloaded: {response.json()}")
        else:
            logger.error(f"Edge Node reload failed with status {response.status_code}: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to communicate with Edge API. Is it running? Error: {e}")


# ----------------------------------------------------------------
# getting data_streamer data 
#-----------------------------------------------------------------
def get_latest_streamed_data() -> pd.DataFrame:
    """
    Loads data using a Sliding Window approach. If a Hard Reset has occurred due 
    to hardware maintenance, it completely drops the 1st-stage baseline data.
    """
    stream_dir = config.BASE_DIR / "stream_data"
    reset_flag_path = config.BASE_DIR / "baseline_reset.txt"
    
    has_reset = os.path.exists(reset_flag_path)
    df_list = []
    
    # --- 1. HANDLE BASELINE OR HARD RESET ---
    if not has_reset:
        logger.info("📊 Loading 1st stage baseline dataset...")
        df_list.append(ml_pipeline.load_dataset())
    else:
        with open(reset_flag_path, "r") as f:
            reset_time = float(f.read().strip())
        logger.warning("🛠️ Hard Reset Active! Structural component change logged. Ignoring 1st stage baseline.")

    # --- 2. GATHER AND FILTER STREAM DATA ---
    list_of_files = glob.glob(str(stream_dir / "*.csv"))
    
    if list_of_files:
        if has_reset:
            # ONLY load stream files created AFTER the physical maintenance happened
            list_of_files = [f for f in list_of_files if os.path.getctime(f) > reset_time]
        
        if list_of_files:
            logger.info(f"🌊 Live stream data detected ({len(list_of_files)} batches). Merging into optimization pool...")
            for file in sorted(list_of_files, key=os.path.getctime):
                df_list.append(pd.read_csv(file))

    # --- 3. SAFETY FALLBACK ---
    if not df_list:
        logger.warning("⚠️ No post-maintenance stream data found yet! Temporarily falling back to baseline to prevent crash.")
        return ml_pipeline.load_dataset()
        
    # Combine historical baseline + all active stream batches
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # --- 4. SLIDING WINDOW IMPLEMENTATION ---
    # Keep a fixed maximum operational window (e.g., 12,000 rows max). 
    # As new rows stream in, old rows naturally get dropped from the top.
    MAX_WINDOW_SIZE = 12000 
    if len(combined_df) > MAX_WINDOW_SIZE:
        logger.info(f"✂️ Sliding Window Active: Trimming oldest records. Keeping newest {MAX_WINDOW_SIZE} rows.")
        combined_df = combined_df.tail(MAX_WINDOW_SIZE).reset_index(drop=True)
        
    return combined_df


def trigger_hard_reset() -> None:
    """Creates a maintenance timestamp flag file to force an immediate baseline reset."""
    reset_flag_path = config.BASE_DIR / "baseline_reset.txt"
    with open(reset_flag_path, "w") as f:
        f.write(str(time.time()))
    logger.info("⚙️ Hard Reset Logged! Old operational baselines successfully archived.")


def clear_hard_reset() -> None:
    """Removes the reset flag file, allowing the pipeline to use the original baseline data again."""
    reset_flag_path = config.BASE_DIR / "baseline_reset.txt"
    if os.path.exists(reset_flag_path):
        os.remove(reset_flag_path)
        logger.info("🔄 Hard Reset cleared. Restoring access to original 1st stage baseline data.")

#--------------------------------------------------------------------------------
#---------------------------------------------------------------------------------


def run_optimization_cycle(force: bool = False) -> None:
    """
    The main self-optimizing loop with built-in model deduplication.
    1. Checks for drift (or forces run).
    2. Re-ingests and preps data.
    3. Trains Candidate via HPO.
    4. Deduplicates: Checks if an identical model already exists in the registry.
    5. Versions, registers, or re-promotes accordingly.
    6. Reloads the edge API.
    """
    logger.info("=== Starting Optimizer Cycle ===")
    
    # 1. Check for Drift
    drift_detected = monitor.run_monitor()
    
    if not drift_detected and not force:
        logger.info("No optimization required at this time.")
        return
        
    if force:
        logger.info("Optimization cycle FORCED by user.")

    # 2. Re-ingest and prep data
    df = get_latest_streamed_data()
    X, y, encoder = ml_pipeline.preprocess_data(df)
    X_train, X_test, y_train, y_test = ml_pipeline.split_data(X, y)
    
    # 3. Train Candidate via Optimization
    best_model, metrics, best_params = optimize_hyperparameters(X_train, y_train, X_test, y_test)
    candidate_roc_auc = metrics.get("roc_auc", 0.0)

    # --- 4. DEDUPLICATION & REGISTRY CHECK (GATE 1) ---
    logger.info("Checking model registry for identical performance matches...")
    
    # Safely query the database to see if this exact performance profile already exists
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT version_number FROM model_versions WHERE ABS(roc_auc - ?) < 1e-5", 
            (candidate_roc_auc,)
        )
        match = cursor.fetchone()
        
    if match:
        logger.warning(
            f"🚫 Retraining Aborted: Calculated ROC-AUC ({candidate_roc_auc:.5f}) "
            f"matches an existing registered version (Version {match['version_number']})."
        )
        logger.info("Skipping file creation and database registration to avoid clutter.")
        logger.info("=== Optimizer Cycle Complete ===")
        return

    # --- 5. EVALUATE PERFORMANCE BEFORE PROMOTION (GATE 2) ---
    next_version = database.get_next_version_number()
    current_deployment = database.get_current_deployment()
    
    # Determine if this model explicitly earns active status
    is_superior = False
    if not current_deployment:
        logger.info("No active deployment found. Champion status granted by default.")
        is_superior = True
    elif candidate_roc_auc > current_deployment["roc_auc"]:
        logger.info(f"🏆 Superior Model Found! ROC-AUC: {candidate_roc_auc:.4f} > Current: {current_deployment['roc_auc']:.4f}")
        is_superior = True

    # --- 6. CONDITIONAL REGISTRATION AND DISK STORAGE ---
    if is_superior:
        # Save production artifact bundle
        bundle_path = ml_pipeline.save_model_bundle(next_version, best_model, encoder, X_train, metrics, best_params)
        
        candidate_id = database.register_model_version(
            version_number=next_version,
            model_path=bundle_path,
            metrics=metrics
        )
        
        # Promote and hot-swap the edge node API live
        database.promote_model_to_production(candidate_id)
        trigger_edge_reload()
        logger.info(f"🚀 Version {next_version} successfully deployed to live edge production.")
    else:
        logger.info(f"📦 Unique but non-improving model. Preserving as an offline 'candidate'.")
        
        # Save data bundle to disk normally
        bundle_path = ml_pipeline.save_model_bundle(next_version, best_model, encoder, X_train, metrics, best_params)
        
        candidate_id = database.register_model_version(
            version_number=next_version,
            model_path=bundle_path,
            metrics=metrics
        )
        
        # Force the database status flag to stay 'candidate' (skips production deployment completely)
        with database.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE model_versions SET status = 'candidate' WHERE id = ?", (candidate_id,))
            conn.commit()
            
        logger.info(f"Saved Version {next_version} into the registry safely flagged as a candidate.")
        
    logger.info("=== Optimizer Cycle Complete ===")

if __name__ == "__main__":
    # Can be run manually with the force flag for demonstration/portfolio purposes
    run_optimization_cycle(force=True)