"""
data_streamer.py

Simulates live IoT telemetry from the factory floor.
Injects artificial data drift (degrading machine conditions) and 
triggers the automated MLOps pipeline to react.
"""

import time
import logging
import pandas as pd
import numpy as np

# Local MLOps imports
import config
import ml_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🏭 STREAMER - %(message)s')
logger = logging.getLogger(__name__)

def generate_live_batch(base_df: pd.DataFrame, batch_size: int = 500, inject_drift: bool = True) -> pd.DataFrame:
    """
    Simulates a new batch of sensor readings by sampling historical data
    and optionally injecting physical degradation (drift) to stress-test the model.
    """
    # 1. Randomly sample 'batch_size' rows from the historical baseline
    live_batch = base_df.sample(n=batch_size, replace=True).copy()
    
    # 2. Simulate new unique IDs for the incoming stream
    max_udi = base_df['UDI'].max() if 'UDI' in base_df.columns else 10000
    live_batch['UDI'] = range(max_udi + 1, max_udi + 1 + batch_size)
    
    # 3. Inject Artificial Concept Drift (Machine Degradation Simulation)
    if inject_drift:
        logger.warning("⚠️ INJECTING DRIFT: Simulating factory floor degradation (High Torque & Tool Wear)...")
        
        torque_col = next((col for col in live_batch.columns if 'Torque' in col), None)
        wear_col = next((col for col in live_batch.columns if 'Tool wear' in col or 'wear' in col.lower()), None)
        
        # Artificially degrade the machines
        if torque_col:
            # Add 15% more torque on average with some random noise
            noise = np.random.normal(loc=1.15, scale=0.05, size=batch_size)
            live_batch[torque_col] = live_batch[torque_col] * noise
            
        if wear_col:
            # Fast-forward tool wear by adding 40-80 minutes of runtime
            live_batch[wear_col] = live_batch[wear_col] + np.random.randint(40, 80, size=batch_size)

        # Force slightly more failures due to the pushed physics
        target = config.TARGET_COLUMN
        if target in live_batch.columns:
            extra_failures = live_batch.sample(frac=0.05).index
            live_batch.loc[extra_failures, target] = 1

    return live_batch

def start_stream(cycles: int = 3, interval_seconds: int = 10):
    """
    Starts the live stream simulation, generating data and triggering the ML Pipeline.
    """
    logger.info("Starting Live Factory Telemetry Stream Simulation...")
    
    if not config.DATASET_PATH.exists():
        logger.error(f"Cannot start stream. Missing base dataset at {config.DATASET_PATH}")
        return

    base_df = pd.read_csv(config.DATASET_PATH)
    
    for cycle in range(1, cycles + 1):
        logger.info(f"\n{'='*50}\n🌊 STREAM CYCLE {cycle}/{cycles}\n{'='*50}")
        
        # Generate 800 rows of drifted data to confuse the Champion model
        incoming_data = generate_live_batch(base_df, batch_size=800, inject_drift=True)
        
        # Save batch to the new STREAM_DIR created in config.py
        stream_path = config.STREAM_DIR / f"live_batch_{int(time.time())}.csv"
        incoming_data.to_csv(stream_path, index=False)
        logger.info(f"Saved incoming telemetry batch: {stream_path}")
        
        # 🚀 TRIGGER THE MLOPS PIPELINE
        logger.info("Triggering EdgeOps ML Pipeline for evaluation...")
        ml_pipeline.run_pipeline(incoming_data=incoming_data)
        
        if cycle < cycles:
            logger.info(f"Waiting {interval_seconds} seconds before next sensor batch...")
            time.sleep(interval_seconds)
            
    logger.info("🏁 Simulation Complete. Pipeline automation successfully tested.")

if __name__ == "__main__":
    # Run 2 simulated streaming cycles, 5 seconds apart
    start_stream(cycles=2, interval_seconds=5)