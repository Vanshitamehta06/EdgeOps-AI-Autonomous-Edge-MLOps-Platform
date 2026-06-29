"""
database.py

Provides the Python API for interacting with the SQLite MLOps registry.
Handles schema initialization, model registration, versioning, and deployment state transitions.
"""

import sqlite3
import logging
from contextlib import contextmanager
from typing import Dict, Optional, Tuple, Any

# Assuming standard project execution from the root directory
from config import SQLITE_DB_PATH, SCHEMA_PATH

# Set up basic logging for the database module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@contextmanager
def get_db_connection():
    """
    Context manager for safe database connections.
    Ensures connections are automatically committed and closed, preventing locks.
    """
    conn = sqlite3.connect(SQLITE_DB_PATH)
    # Configure row factory to return dictionaries instead of tuples for cleaner API responses
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    """
    Initializes the database schema using the schema.sql file.
    Idempotent operation; safe to run multiple times.
    """
    schema_path = SCHEMA_PATH
    
    if not schema_path.exists():
        logger.error(f"Schema file not found at {schema_path}")
        raise FileNotFoundError(f"Missing schema file: {schema_path}")

    with open(schema_path, "r") as f:
        schema_script = f.read()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executescript(schema_script)
        conn.commit()
    logger.info("Database schema initialized successfully.")

def get_next_version_number() -> int:
    """
    Determines the next sequential model version number based on existing registry data.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(version_number) as max_version FROM model_versions")
        result = cursor.fetchone()
        
    # If no models exist, start at version 1
    if result["max_version"] is None:
        return 1
    return result["max_version"] + 1

def register_model_version(version_number: int, model_path: str, metrics: Dict[str, float]) -> int:
    """
    Inserts a newly trained model's metadata and performance metrics into the registry.
    """
    query = """
        INSERT INTO model_versions 
        (version_number, model_path, accuracy, precision_score, recall_score, f1_score, roc_auc)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        version_number,
        model_path,
        metrics.get("accuracy", 0.0),
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1", 0.0),
        metrics.get("roc_auc", 0.0)
    )

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, values)
        conn.commit()
        model_id = cursor.lastrowid
        
    logger.info(f"Registered model version {version_number} with ID {model_id}.")
    return model_id

def promote_model_to_production(model_id: int):
    """
    Updates the deployment_state to point to the newly compiled model,
    making it the active production endpoint.
    """
    import sqlite3
    import config

    conn = sqlite3.connect(config.SQLITE_DB_PATH, timeout=30.0)
    cursor = conn.cursor()

    # 1. Fetch the currently active model to demote it to 'stable_model_id' (fallback)
    cursor.execute("SELECT current_model_id FROM deployment_state WHERE id = 1")
    row = cursor.fetchone()
    stable_id = row[0] if row else None
    
    # 2. THE FIX: Use 'updated_at' to match the deployment_state schema
    update_query = """
        INSERT INTO deployment_state (id, current_model_id, stable_model_id, updated_at)
        VALUES (1, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(id) DO UPDATE SET 
            current_model_id = excluded.current_model_id,
            stable_model_id = excluded.stable_model_id,
            updated_at = CURRENT_TIMESTAMP;
    """
    cursor.execute(update_query, (model_id, stable_id))
    
    # 3. Update the statuses in the model_versions table
    # (This table DOES use last_updated according to your schema)
    cursor.execute("UPDATE model_versions SET status = 'archived' WHERE status = 'active'")
    cursor.execute("UPDATE model_versions SET status = 'active', last_updated = CURRENT_TIMESTAMP WHERE id = ?", (model_id,))
    
    conn.commit()
    conn.close()

def get_current_deployment() -> Optional[Dict[str, Any]]:
    """
    Retrieves the fully joined metadata of the currently deployed active model.
    """
    query = """
        SELECT mv.* FROM deployment_state ds
        JOIN model_versions mv ON ds.current_model_id = mv.id
        WHERE ds.id = 1
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()
        
    return dict(result) if result else None

def rollback_deployment() -> bool:
    """
    Reverts the active production model to the stable fallback model.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT stable_model_id FROM deployment_state WHERE id = 1")
        state = cursor.fetchone()
        
        if not state or state["stable_model_id"] is None:
            logger.warning("Rollback failed: No stable model ID found.")
            return False
            
        stable_id = state["stable_model_id"]
        
        # The old stable becomes current, and we set stable to NULL 
        # (or we could keep a deeper history, but for this setup, we avoid a loop)
        update_query = """
            UPDATE deployment_state 
            SET current_model_id = ?, stable_model_id = NULL, last_updated = CURRENT_TIMESTAMP
            WHERE id = 1
        """
        cursor.execute(update_query, (stable_id,))
        conn.commit()
        
    logger.info(f"Rolled back production to model ID {stable_id}.")
    return True

init_db()