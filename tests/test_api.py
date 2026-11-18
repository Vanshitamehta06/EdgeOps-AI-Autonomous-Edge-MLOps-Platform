"""
FastAPI Integration Test Suite.
Verifies all REST API endpoints, edge inference payload validation,
role-based Basic Authentication on safety-critical endpoints (/rollback, /hard_reset, /thresholds),
and dashboard data integrity.
"""

import base64
import pytest
from fastapi.testclient import TestClient
from api import app
import database
import config

client = TestClient(app)

# Helper for Basic Auth header
ADMIN_USER = config.EDGE_ADMIN_USER
ADMIN_PASS = config.EDGE_ADMIN_PASSWORD
AUTH_TOKEN = base64.b64encode(f"{ADMIN_USER}:{ADMIN_PASS}".encode()).decode()
AUTH_HEADERS = {"Authorization": f"Basic {AUTH_TOKEN}"}


def test_health_endpoint():
    """Verify system health endpoint returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert "timestamp" in data


def test_dashboard_state_endpoint():
    """Verify deployment state singleton is queried."""
    response = client.get("/dashboard/state")
    assert response.status_code == 200
    data = response.json()
    assert "current_model_id" in data
    assert "stable_model_id" in data


def test_dashboard_registry_endpoint():
    """Verify model registry returns version list."""
    response = client.get("/dashboard/registry")
    assert response.status_code == 200
    models = response.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    assert "version_number" in models[0]


def test_predict_endpoint_valid():
    """Verify edge inference returns valid prediction and root cause diagnostics."""
    payload = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "root_cause" in data
    assert data["prediction"] in (0, 1)


def test_auth_protection_on_rollback():
    """Verify /rollback is rejected with 401 without credentials and accepted with valid Basic Auth."""
    # Ensure there is a valid rollback target in deployment_state
    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM model_versions ORDER BY version_number ASC LIMIT 2")
        rows = cursor.fetchall()
        if len(rows) >= 2:
            database.promote_model_to_production(rows[0]["id"])
            database.promote_model_to_production(rows[1]["id"])

    # Without credentials -> 401
    resp_unauth = client.post("/rollback")
    assert resp_unauth.status_code == 401

    # With credentials -> 200
    resp_auth = client.post("/rollback", headers=AUTH_HEADERS)
    assert resp_auth.status_code == 200
    assert resp_auth.json().get("status") == "success"


def test_auth_protection_on_hard_reset():
    """Verify /hard_reset is rejected with 401 without credentials and accepted with valid Basic Auth."""
    # Without credentials -> 401
    resp_unauth = client.post("/hard_reset")
    assert resp_unauth.status_code == 401

    # With credentials -> 200
    resp_auth = client.post("/hard_reset", headers=AUTH_HEADERS)
    assert resp_auth.status_code == 200
    assert resp_auth.json().get("status") == "success"


def test_thresholds_get_and_post():
    """Verify threshold query and update with authentication."""
    # GET thresholds is public
    resp_get = client.get("/thresholds")
    assert resp_get.status_code == 200
    curr_thresh = resp_get.json()
    assert isinstance(curr_thresh, dict)

    # POST thresholds requires auth
    update_payload = {"Air temperature [K]": {"mean": 300.0, "std": 2.0}}
    resp_unauth = client.post("/thresholds", json=update_payload)
    assert resp_unauth.status_code == 401

    resp_auth = client.post("/thresholds", json=curr_thresh, headers=AUTH_HEADERS)
    assert resp_auth.status_code == 200
    assert resp_auth.json().get("status") == "success"


def test_alerts_endpoints():
    """Verify alerts querying and clearing."""
    resp_alerts = client.get("/alerts")
    assert resp_alerts.status_code == 200
    data = resp_alerts.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)

    resp_clear = client.post("/alerts/clear")
    assert resp_clear.status_code == 200
    assert resp_clear.json().get("status") == "success"
