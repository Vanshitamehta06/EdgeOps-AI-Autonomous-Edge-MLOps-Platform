"""
Locust load test for the EdgeOps AI Inference API.
Simulates concurrent industrial telemetry streams pushing sensor payloads
at high frequency to evaluate edge throughput, latency percentiles, and error rate.

Run with:
    locust -f tests/locustfile.py --headless -u 50 -r 10 --run-time 30s --host http://127.0.0.1:8000
"""

import random
from locust import HttpUser, task, between


class IndustrialSensorStreamUser(HttpUser):
    # Simulated sensor publish interval (100ms - 500ms)
    wait_time = between(0.1, 0.5)

    @task(10)
    def post_telemetry_inference(self):
        """High-frequency telemetry inference request."""
        sample_payload = {
            "Type": random.choice(["L", "M", "H"]),
            "Air temperature [K]": round(random.uniform(295.0, 305.0), 2),
            "Process temperature [K]": round(random.uniform(305.0, 315.0), 2),
            "Rotational speed [rpm]": round(random.uniform(1300.0, 1800.0), 1),
            "Torque [Nm]": round(random.uniform(20.0, 60.0), 2),
            "Tool wear [min]": round(random.uniform(0.0, 240.0), 1)
        }
        self.client.post("/predict", json=sample_payload, name="/predict")

    @task(3)
    def get_dashboard_state(self):
        """Dashboard telemetry polling (1s interval)."""
        self.client.get("/dashboard/state", name="/dashboard/state")

    @task(2)
    def get_live_alerts(self):
        """Dashboard alert polling."""
        self.client.get("/alerts", name="/alerts")

    @task(1)
    def get_drift_status(self):
        """Evidently/KS drift polling."""
        self.client.get("/dashboard/drift", name="/dashboard/drift")
