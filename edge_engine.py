"""
edge_engine.py

Simulates the Edge AI inference engine.
Responsible for dynamically loading the active ONNX model from the MLOps registry,
managing memory, and executing highly optimized CPU-based predictions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort

# Local imports
import database
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EdgeEngine:
    """
    A persistent state engine for edge inference. 
    Maintains the loaded ONNX session in memory to prevent disk I/O bottlenecks during prediction.
    """
    
    def __init__(self):
        self.session: Optional[ort.InferenceSession] = None
        self.encoder: Any = None
        self.metadata: Dict[str, Any] = {}
        self.current_version: Optional[int] = None
        self.input_name: str = ""
        
        # Automatically attempt to load the current production model on startup
        self.reload_model()

    def reload_model(self) -> bool:
        """
        Queries the database for the current deployed model and loads its bundle into memory.
        This is called on startup and can be triggered by the API when a new model is promoted.
        """
        logger.info("Checking registry for active production model...")
        deployment = database.get_current_deployment()
        
        if not deployment:
            logger.warning("No active deployment found in the registry. Engine is idling.")
            return False
            
        model_path = Path(deployment["model_path"])
        version_number = deployment["version_number"]
        
        if self.current_version == version_number:
            logger.info(f"Model v{version_number} is already loaded. Skipping reload.")
            return True

        logger.info(f"Loading model bundle for version {version_number} from {model_path}...")
        
        try:
            # 1. Load Metadata
            with open(model_path / "metadata.json", "r") as f:
                self.metadata = json.load(f)
                
            # 2. Load Label Encoder (required for the 'Type' categorical feature)
            self.encoder = joblib.load(model_path / "label_encoder.pkl")
            
            # 3. Load ONNX Session (Strictly CPU execution provider)
            onnx_path = str(model_path / "model.onnx")
            self.session = ort.InferenceSession(
                onnx_path, 
                providers=["CPUExecutionProvider"]
            )
            
            # Cache the expected ONNX input node name (usually 'float_input')
            self.input_name = self.session.get_inputs()[0].name
            self.current_version = version_number
            
            logger.info(f"Successfully loaded Edge Engine with v{version_number}.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model bundle: {e}")
            return False

    def predict(self, raw_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes incoming raw telemetry, applies preprocessing, and executes ONNX inference.
        """
        if self.session is None:
            raise RuntimeError("Edge Engine has no model loaded. Cannot execute prediction.")

        # 1. Validate Schema
        expected_features = self.metadata.get("feature_columns", [])
        missing_features = [f for f in expected_features if f not in raw_telemetry]
        if missing_features:
            raise ValueError(f"Telemetry payload missing required features: {missing_features}")

        # 2. Preprocess Data
        # Convert dictionary to DataFrame to ensure exact column ordering as trained
        df = pd.DataFrame([raw_telemetry])
        
        # Apply encoding to the categorical 'Type' column
        if "Type" in df.columns:
            try:
                df["Type"] = self.encoder.transform(df["Type"])
            except ValueError as e:
                # Handle unseen categorical labels gracefully at the edge
                logger.warning(f"Unseen 'Type' encountered: {df['Type'].iloc[0]}. Defaulting to mode.")
                # For safety in production edge, fallback to a known safe value (e.g., encoded 0)
                df["Type"] = 0 
                
        # Ensure exact column order and convert to float32 numpy array (required by ONNX)
        X_infer = df[expected_features].astype(np.float32).values

        # 3. Execute ONNX Inference
        # session.run returns a list of outputs: [predicted_labels, probabilities_dictionary]
        ort_inputs = {self.input_name: X_infer}
        ort_outs = self.session.run(None, ort_inputs)
        
        # Extract binary prediction (0 or 1)
        prediction = int(ort_outs[0][0])
        
        # Extract probability of failure (Class 1)
        # ONNX ZipMap outputs probabilities as a list of dicts: [{0: prob, 1: prob}]
        probabilities = ort_outs[1][0]
        probability_of_failure = float(probabilities.get(1, 0.0))

        # 4. Format Output
        return {
            "model_version": self.current_version,
            "prediction": prediction,
            "probability": probability_of_failure,
            "threshold_used": config.PREDICTION_THRESHOLD,
            "is_anomaly": probability_of_failure >= config.PREDICTION_THRESHOLD
        }

# Instantiate a global singleton engine to be imported by the API later
engine = EdgeEngine()