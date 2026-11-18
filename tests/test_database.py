"""
Unit and integration tests for EdgeOps SQLite model registry and deployment state.
"""

import sqlite3
import pytest
from pathlib import Path
import database
import config


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    """Use a temporary database for test isolation."""
    test_db = tmp_path / "test_edgeops.db"
    monkeypatch.setattr(database, "SQLITE_DB_PATH", test_db)
    monkeypatch.setattr(config, "SQLITE_DB_PATH", test_db)
    database.init_db()
    yield test_db


def test_schema_and_tables_exist():
    """Verify that all core MLOps tables exist in the initialized schema."""
    with sqlite3.connect(database.SQLITE_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall()]
        assert "model_versions" in tables
        assert "deployment_state" in tables
        assert "system_metrics" in tables


def test_model_registration_and_promotion():
    """Test registering candidate models and promoting them to production."""
    metrics_v1 = {"accuracy": 0.95, "precision": 0.80, "recall": 0.82, "f1": 0.81, "roc_auc": 0.96}
    m1_id = database.register_model_version(
        version_number=1,
        model_path="models/version_0001/model.onnx",
        metrics=metrics_v1
    )
    assert m1_id is not None
    assert m1_id >= 1

    metrics_v2 = {"accuracy": 0.97, "precision": 0.85, "recall": 0.88, "f1": 0.86, "roc_auc": 0.98}
    m2_id = database.register_model_version(
        version_number=2,
        model_path="models/version_0002/model.onnx",
        metrics=metrics_v2
    )
    assert m2_id > m1_id

    # Promote model 1 to production
    database.promote_model_to_production(m1_id)
    deployment = database.get_current_deployment()
    assert deployment is not None
    assert deployment["version_number"] == 1
    assert deployment["status"] == "active"

    # Promote model 2: model 1 becomes stable, model 2 becomes active
    database.promote_model_to_production(m2_id)
    deployment2 = database.get_current_deployment()
    assert deployment2["version_number"] == 2
    assert deployment2["status"] == "active"

    with database.get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT current_model_id, stable_model_id FROM deployment_state WHERE id = 1")
        state = cursor.fetchone()
        assert state["current_model_id"] == m2_id
        assert state["stable_model_id"] == m1_id


def test_atomic_rollback():
    """Test atomic rollback from active model to stable fallback."""
    m1_id = database.register_model_version(1, "models/v1.onnx", {"f1": 0.80})
    m2_id = database.register_model_version(2, "models/v2.onnx", {"f1": 0.85})
    database.promote_model_to_production(m1_id)
    database.promote_model_to_production(m2_id)

    # Rollback deployment
    ok = database.rollback_deployment()
    assert ok is True

    # Active model should now be m1
    current = database.get_current_deployment()
    assert current["id"] == m1_id
    assert current["status"] == "active"

    # Secondary rollback should fail gracefully since stable_model_id is now None
    assert database.rollback_deployment() is False


def test_score_exists_and_version_increment():
    """Test score deduplication and auto-increment version calculation."""
    assert database.get_next_version_number() == 1
    database.register_model_version(1, "models/v1.onnx", {"f1": 0.8752})
    assert database.get_next_version_number() == 2

    # Verify score exists check
    assert database.check_score_exists(0.8752) is True
    assert database.check_score_exists(0.5000) is False
