"""
monitor_realtime.py
Calculates safe operational limits from baseline data and evaluates real-time streams.
"""
import os
import json
import time
import logging
from datetime import datetime
import pandas as pd

import config

logger = logging.getLogger(__name__)

THRESHOLDS_FILE = config.BASE_DIR / "sensor_thresholds.json"
_active_alerts = []

def calculate_and_save_thresholds(df: pd.DataFrame):
    """Calculates normal distributions from 'good data' to generate 3 alert tiers."""
    skip_cols = set(config.COLUMNS_TO_DROP + [config.TARGET_COLUMN])
    sensor_cols = [c for c in df.columns if c not in skip_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    thresholds = {}
    for col in sensor_cols:
        mean_val = float(df[col].mean())
        std_val = float(df[col].std())
        if std_val == 0: std_val = 1e-4
            
        thresholds[col] = {
            "mean": mean_val, "std": std_val,
            "low_lower": mean_val - 2 * std_val, "low_upper": mean_val + 2 * std_val,
            "mid_lower": mean_val - 3 * std_val, "mid_upper": mean_val + 3 * std_val,
            "high_lower": mean_val - 4 * std_val, "high_upper": mean_val + 4 * std_val,
        }
        
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(thresholds, f, indent=4)
    logger.info(f"🎯 Dynamic operational limits set for {len(thresholds)} sensors.")

def load_thresholds():
    """Loads the active thresholds from JSON."""
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_manual_thresholds(new_thresholds: dict):
    """Allows operators to manually override the auto-calculated thresholds."""
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(new_thresholds, f, indent=4)
    logger.info("🛠️ Manual threshold overrides applied and saved.")

def get_active_alerts():
    return _active_alerts

def clear_alerts():
    global _active_alerts
    _active_alerts = []

def evaluate_realtime_row(row: dict) -> list:
    """Evaluates an incoming streaming telemetry record against dynamic limits."""
    global _active_alerts
    thresholds = load_thresholds()
    if not thresholds: return []
        
    new_alerts = []
    for sensor, limits in thresholds.items():
        if sensor in row:
            val = float(row[sensor])
            level, msg = None, ""
            
            if val > limits["high_upper"] or val < limits["high_lower"]:
                level, msg = "High", f"{sensor} critically out of bounds: {val:.2f}"
            elif val > limits["mid_upper"] or val < limits["mid_lower"]:
                level, msg = "Mid", f"{sensor} moderately out of bounds: {val:.2f}"
            elif val > limits["low_upper"] or val < limits["low_lower"]:
                level, msg = "Low", f"{sensor} slightly out of bounds: {val:.2f}"
                
            if level:
                alert = {
                    "id": int(time.time() * 1000) + len(new_alerts),
                    "level": level, "location": sensor, "message": msg,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                new_alerts.append(alert)
                _active_alerts.insert(0, alert)
                _active_alerts = _active_alerts[:50]  # Keep history max 50
                
    return new_alerts