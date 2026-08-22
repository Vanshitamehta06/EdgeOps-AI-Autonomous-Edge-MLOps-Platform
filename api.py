"""
api.py

Provides the REST API for the EdgeOps AI platform.
Exposes endpoints for health checks, live telemetry inference, and hot-swapping models.
Now features CORS handling and dedicated data/control endpoints for the React dashboard.
"""

import json
import logging
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
import config
import database
import ml_pipeline
import optimizer
import data_streamer
from edge_engine import engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EdgeOps AI API",
    description="Simulated Edge Environment for Predictive Maintenance Inference",
    version="1.0.0"
)

# ---------------------------------------------------------
# CORS MIDDLEWARE (Crucial for React Frontend Integration)
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your React domain (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# PYDANTIC SCHEMAS
# ---------------------------------------------------------

class TelemetryPayload(BaseModel):
    """Defines the strict schema for incoming machine telemetry."""
    Type: str = Field(..., description="Quality variant of the product (e.g., L, M, H)")
    Air_temperature_K: float = Field(..., alias="Air temperature [K]")
    Process_temperature_K: float = Field(..., alias="Process temperature [K]")
    Rotational_speed_rpm: float = Field(..., alias="Rotational speed [rpm]")
    Torque_Nm: float = Field(..., alias="Torque [Nm]")
    Tool_wear_min: float = Field(..., alias="Tool wear [min]")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "Type": "M",
                "Air temperature [K]": 298.1,
                "Process temperature [K]": 308.6,
                "Rotational speed [rpm]": 1551,
                "Torque [Nm]": 42.8,
                "Tool wear [min]": 0
            }
        }

class PredictionResponse(BaseModel):
    """Schema for the API prediction response."""
    model_version: int
    prediction: int
    probability: float
    threshold_used: float
    is_anomaly: bool
    timestamp: str

# ---------------------------------------------------------
# BACKGROUND TASKS
# ---------------------------------------------------------

def log_telemetry(payload: dict, response: dict) -> None:
    """Appends incoming requests and predictions to a local JSONL file for drift evaluation."""
    log_file = config.LOGS_DIR / "inference_logs.jsonl"
    
    log_entry = {
        "timestamp": response["timestamp"],
        "model_version": response["model_version"],
        "Type": payload["Type"],
        "Air temperature [K]": payload["Air_temperature_K"],
        "Process temperature [K]": payload["Process_temperature_K"],
        "Rotational speed [rpm]": payload["Rotational_speed_rpm"],
        "Torque [Nm]": payload["Torque_Nm"],
        "Tool wear [min]": payload["Tool_wear_min"],
        "prediction": response["prediction"],
        "probability": response["probability"]
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

# ---------------------------------------------------------
# CORE INFERENCE & EDGE ENDPOINTS
# ---------------------------------------------------------

@app.get("/health")
def health_check():
    """Returns the operational status of the edge node and the loaded model."""
    status = "healthy" if engine.session else "degraded"
    return {
        "status": status,
        "active_model_version": engine.current_version,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TelemetryPayload, background_tasks: BackgroundTasks):
    """Executes live, edge inference and non-blockingly tracks telemetry transactions."""
    if engine.session is None:
        raise HTTPException(
            status_code=503, 
            detail="Edge Engine has no active model loaded. Please deploy a model first."
        )

    try:
        raw_telemetry = payload.model_dump(by_alias=True)
        result = engine.predict(raw_telemetry)
        result["timestamp"] = datetime.utcnow().isoformat()
        
        background_tasks.add_task(log_telemetry, payload.model_dump(), result)
        return result

    except ValueError as ve:
        logger.error(f"Validation error during prediction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during inference.")

@app.post("/reload")
def reload_model():
    """Forces the Edge Engine to hot-swap the active runtime engine model."""
    success = engine.reload_model()
    if success:
        return {"message": f"Successfully reloaded. Active version: {engine.current_version}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model from registry.")

# ---------------------------------------------------------
# REACT DASHBOARD DATA READ ENDPOINTS
# ---------------------------------------------------------

@app.get("/dashboard/registry")
def get_registry():
    """Fetches all registered version rows for the React artifact registry table."""
    try:
        with database.get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM model_versions ORDER BY version_number DESC", conn)
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Failed to fetch registry data: {e}")
        return []

@app.get("/dashboard/state")
def get_state():
    """Fetches active/stable model metrics configurations for React dashboard cards."""
    try:
        with database.get_db_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM deployment_state WHERE id = 1", conn)
        return df.iloc[0].to_dict() if not df.empty else {}
    except Exception as e:
        logger.error(f"Failed to fetch deployment state: {e}")
        return {}

@app.get("/dashboard/drift")
def get_drift_state():
    """Exposes current data drift tracking from Evidently AI to the React UI."""
    drift_file = config.BASE_DIR / "drift_state.json"
    if drift_file.exists():
        try:
            with open(drift_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading drift state file: {e}")
    return {"drift_detected": False, "drift_share": 0.0, "timestamp": ""}

# ---------------------------------------------------------
# REACT DASHBOARD PIPELINE PIPES (Action Triggers)
# ---------------------------------------------------------

@app.post("/run_pipeline")
def trigger_pipeline():
    """Executes initial schema assembly and base model training sequence."""
    try:
        database.init_db()
        ml_pipeline.run_pipeline()
        return {"status": "success", "message": "Pipeline completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.post("/simulate_stream")
def trigger_live_stream():
    """Injects high-velocity, mutated machine data structures to evaluate gating limits."""
    try:
        data_streamer.start_stream(cycles=1, interval_seconds=0)
        return {"status": "success", "message": "Telemetry stream cycle dispatched"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telemetry streaming failed: {str(e)}")

@app.post("/force_optimization")

def trigger_optimization():
    """Bypasses scheduling configurations to aggressively optimize model variants."""
    try:
        optimizer.run_optimization_cycle(force=True)
        return {"status": "success", "message": "Optimization completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forced optimization run failed: {str(e)}")

@app.post("/hard_reset")
def trigger_api_hard_reset():
    """Logs a hardware maintenance event and triggers a data baseline reset."""
    try:
        optimizer.trigger_hard_reset()
        return {"status": "success", "message": "Hard reset logged successfully. AI is tracking new hardware modifications."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hard reset failed: {str(e)}")

@app.post("/rollback")
def trigger_rollback():
    """Executes urgent model rollback workflows to fall back onto stable profiles."""
    try:
        success = database.rollback_deployment()
        if success:
            optimizer.trigger_edge_reload()
            return {"status": "success", "message": "Rollback successful and Edge Node hot-swapped"}
        else:
            raise HTTPException(status_code=400, detail="Rollback failed. No stable historical model configuration available.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rollback operation failed: {str(e)}")
