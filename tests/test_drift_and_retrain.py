"""
Unit and integration tests for Statistical Drift Detection and Automated Retraining Triggers.
Verifies KS-test, PSI calculation, clean stream stability, and drift alert thresholds.
"""

import numpy as np
import pandas as pd
import pytest
import monitor
import config


@pytest.fixture
def clean_and_drifted_data():
    """Generates synthetic baseline and drifted industrial telemetry distributions."""
    np.random.seed(42)
    n = 200
    # Baseline
    ref_df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300.0, 2.0, n),
        "Process temperature [K]": np.random.normal(310.0, 1.5, n),
        "Rotational speed [rpm]": np.random.normal(1538.0, 50.0, n),
        "Torque [Nm]": np.random.normal(40.0, 5.0, n),
        "Tool wear [min]": np.random.uniform(0, 200, n),
        "Type": np.random.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    })

    # Clean current (same distribution)
    clean_cur_df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300.0, 2.0, n),
        "Process temperature [K]": np.random.normal(310.0, 1.5, n),
        "Rotational speed [rpm]": np.random.normal(1538.0, 50.0, n),
        "Torque [Nm]": np.random.normal(40.0, 5.0, n),
        "Tool wear [min]": np.random.uniform(0, 200, n),
        "Type": np.random.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    })

    # Severe drifted current (Torque +3 SD, Speed shifted, Process Temp elevated)
    drifted_cur_df = pd.DataFrame({
        "Air temperature [K]": np.random.normal(300.0, 2.0, n),
        "Process temperature [K]": np.random.normal(318.0, 1.5, n),     # Drifted
        "Rotational speed [rpm]": np.random.normal(1350.0, 50.0, n),    # Drifted
        "Torque [Nm]": np.random.normal(55.0, 5.0, n),                  # Drifted (+3 SD)
        "Tool wear [min]": np.random.uniform(0, 200, n),
        "Type": np.random.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    })

    return ref_df, clean_cur_df, drifted_cur_df


def test_psi_calculation_identical():
    """Verify that identical distributions have near-zero PSI."""
    arr = np.random.normal(100, 15, 500)
    psi = monitor.calculate_psi(arr, arr)
    assert psi < 0.05, f"PSI for identical distribution should be ~0, got {psi}"


def test_psi_calculation_shifted():
    """Verify that a shifted distribution exceeds the standard PSI threshold (0.25)."""
    ref = np.random.normal(100, 15, 500)
    shifted = np.random.normal(130, 15, 500)
    psi = monitor.calculate_psi(ref, shifted)
    assert psi > 0.25, f"PSI for shifted distribution should exceed 0.25, got {psi}"


def test_clean_stream_no_false_alarm(clean_and_drifted_data):
    """Verify that unshifted stream does not trigger the 30% drift threshold."""
    ref_df, clean_cur_df, _ = clean_and_drifted_data
    report = monitor.analyze_data_drift(ref_df, clean_cur_df)
    
    assert report["drift_detected"] is False
    assert report["drift_share"] < config.DRIFT_THRESHOLD


def test_drift_stream_triggers_retrain_threshold(clean_and_drifted_data):
    """Verify that controlled multi-feature drift triggers dataset_drift == True."""
    ref_df, _, drifted_cur_df = clean_and_drifted_data
    report = monitor.analyze_data_drift(ref_df, drifted_cur_df)

    assert report["drift_detected"] is True
    assert report["drift_share"] >= config.DRIFT_THRESHOLD
    assert "Torque [Nm]" in report["drifted_features"]
    assert "Process temperature [K]" in report["drifted_features"]
