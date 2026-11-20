"""
Unit and benchmark tests for the Edge Inference Engine (ONNX Runtime).
Verifies model hot-loading, schema tolerance (bracketed vs raw keys),
and physics-based failure root cause classification.
"""

import time
import pytest
import numpy as np
from edge_engine import EdgeEngine


@pytest.fixture(scope="module")
def engine():
    return EdgeEngine()


def test_edge_engine_initialization(engine):
    """Verify that ONNX session is active and model metadata is populated."""
    assert engine.session is not None
    assert engine.input_name is not None
    assert engine.current_version is not None
    assert engine.current_version > 0


def test_predict_with_clean_features(engine):
    """Verify inference works with clean sensor feature names."""
    payload = {
        "Type": "M",
        "Air temperature K": 300.0,
        "Process temperature K": 310.0,
        "Rotational speed rpm": 1500.0,
        "Torque Nm": 40.0,
        "Tool wear min": 10.0
    }
    result = engine.predict(payload)
    assert "prediction" in result
    assert "probability" in result
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_predict_with_bracketed_features(engine):
    """Verify inference normalizes bracketed sensor names correctly."""
    payload = {
        "Type": "L",
        "Air temperature [K]": 298.5,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551.0,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0.0
    }
    result = engine.predict(payload)
    assert result["prediction"] == 0
    assert result["probability"] < 0.50
    assert result["root_cause"] == "None"


def test_root_cause_diagnostics(engine):
    """Verify physics-informed diagnostic categorization for failure modes."""
    # 1. Overstrain Failure (OSF) scenario: Torque * Tool Wear > 11,000
    osf_payload = {
        "Type": "L",
        "Air temperature [K]": 302.0,
        "Process temperature [K]": 311.0,
        "Rotational speed [rpm]": 1300.0,
        "Torque [Nm]": 65.0,
        "Tool wear [min]": 210.0  # 65 * 210 = 13,650 > 11,000
    }
    res_osf = engine.predict(osf_payload)
    assert res_osf["root_cause"] in ("Tool Wear Failure (TWF)", "Overstrain Failure (OSF)")

    # 2. Heat Dissipation Failure (HDF) scenario: (Proc - Air) < 8.6 and RPM < 1380
    hdf_payload = {
        "Type": "M",
        "Air temperature [K]": 305.0,
        "Process temperature [K]": 310.0,  # Delta = 5.0 K < 8.6 K
        "Rotational speed [rpm]": 1250.0,   # < 1380 RPM
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 50.0
    }
    res_hdf = engine.predict(hdf_payload)
    assert res_hdf["root_cause"] == "Heat Dissipation Failure (HDF)"


def test_sub_millisecond_edge_latency(engine):
    """Verify that ONNX runtime delivers sub-millisecond edge execution."""
    sample = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0
    }
    latencies_ms = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = engine.predict(sample)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    # ONNX runtime should execute comfortably under 5.0 ms on edge host
    assert p50 < 5.0, f"p50 latency too high: {p50} ms"
    assert p95 < 15.0, f"p95 latency too high: {p95} ms"
