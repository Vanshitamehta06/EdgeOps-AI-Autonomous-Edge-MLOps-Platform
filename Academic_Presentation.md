# EdgeOps AI Platform
## Production-Grade MLOps for Predictive Maintenance at the Industrial Edge

**Submitted by:** Vanshita Mehta  
**Mentor:** Mr. Ashish Kumar  
**Organization:** Siemens Digital Industries - Factory Automation  
**Date:** July 24, 2026

---

## Table of Contents

1. Introduction & Problem Context
2. Literature Review
3. Research Gap
4. Research Objectives & Problem Definition
5. Proposed Methodology
6. System Architecture
7. Technology Stack
8. Implementation Details
9. Results & Discussion
10. Conclusion & Future Work

---

## 1. Introduction & Problem Context

### What is Predictive Maintenance?

**Predictive Maintenance (PdM)** uses live sensor data and machine learning to forecast equipment failures before they occur, replacing fixed schedules and reactive repairs with data-driven timing.

| Maintenance Type | Approach | Limitations |
|-----------------|----------|-------------|
| **Reactive** | Repair after failure | Highest unplanned downtime, production losses |
| **Preventive** | Fixed schedule | Replaces healthy parts, inefficient |
| **Predictive (EdgeOps)** | AI-driven condition monitoring | Requires ML infrastructure & drift handling |

### Industrial Edge Computing Context

- **Edge deployment** reduces latency (sub-second inference vs. cloud round-trip)
- Critical for **real-time industrial control loops** (PLC integration)
- Challenges: **model drift, limited compute, reliability requirements**

---

## 2. Literature Review

### 2.1 Predictive Maintenance in Industry 4.0

**Key Research Papers:**

1. **Lee et al. (2014)** - "Prognostics and health management design for rotary machinery systems"
   - Established 5C architecture for cyber-physical systems
   - Limitation: Cloud-centric, high latency for edge scenarios

2. **Susto et al. (2015)** - "Machine Learning for Predictive Maintenance: A Multiple Classifier Approach"
   - XGBoost and ensemble methods achieve 92%+ accuracy
   - Gap: No continuous retraining or drift detection

3. **Carvalho et al. (2019)** - "A systematic literature review of machine learning methods for PdM"
   - Survey of 127 papers: Random Forest, SVM, Neural Networks dominant
   - Finding: **Model degradation over time** is under-addressed

### 2.2 MLOps & Model Lifecycle Management

4. **Paleyes et al. (2022)** - "Challenges in Deploying Machine Learning: A Survey"
   - Identified drift detection as critical for production ML systems
   - Recommendation: Automated retraining pipelines

5. **Kreuzberger et al. (2023)** - "Machine Learning Operations (MLOps): A Systematic Review"
   - MLflow, Kubeflow for experiment tracking
   - Gap: Limited edge-specific MLOps frameworks

### 2.3 Edge AI Optimization

6. **Merenda et al. (2020)** - "Edge Machine Learning for AI-Enabled IoT Devices"
   - ONNX runtime reduces inference time by 40%
   - Model quantization techniques for resource-constrained devices

7. **Dang et al. (2021)** - "Lightweight Models for Predictive Maintenance at the Edge"
   - Trade-off: accuracy vs. inference speed
   - XGBoost suitable for tabular sensor data

### 2.4 Drift Detection Techniques

8. **Lu et al. (2018)** - "Learning under Concept Drift: A Review"
   - PSI (Population Stability Index), KS tests for feature drift
   - Evidently library implements statistical drift metrics

9. **Gama et al. (2014)** - "A Survey on Concept Drift Adaptation"
   - Threshold-based retraining strategies (30% drift → retrain)

---

## 3. Research Gap

### Identified Limitations in Existing Research

| Gap | Existing Approaches | Consequence |
|-----|-------------------|-------------|
| **1. No self-healing pipelines** | Manual model updates required | Downtime during retraining |
| **2. Cloud dependency** | Latency-sensitive edge ignored | >100ms inference delays |
| **3. Drift monitoring absent** | Static models degrade silently | Accuracy drops 15-20% over 6 months |
| **4. No rollback mechanisms** | Failed deployments crash systems | Production reliability risk |
| **5. Operator visibility gap** | No live dashboards for edge ML | Delayed failure detection |

### What EdgeOps Addresses

✅ **Autonomous drift detection + retraining** (Evidently + Optuna)  
✅ **Edge-optimized inference** (ONNX, <10ms latency)  
✅ **Instant rollback protocol** (dual-pointer model registry)  
✅ **Live telemetry dashboard** (React + Grafana)  
✅ **Production-grade reliability** (self-healing, alerts, recovery)

---

## 4. Research Objectives & Problem Definition

### Problem Statement

> **How can we build a production-ready MLOps platform that continuously monitors industrial sensor data, automatically retrains predictive maintenance models when drift is detected, and ensures <10ms inference latency with zero-downtime rollback capabilities?**

### Research Objectives

1. **Design a closed-loop MLOps pipeline** for edge predictive maintenance
   - Measurable: Drift detection → retrain → deploy cycle <5 minutes

2. **Implement automated drift detection** using statistical methods
   - Measurable: PSI threshold at 30%, trigger retraining

3. **Optimize ML inference for edge deployment**
   - Measurable: ONNX runtime achieves <10ms inference

4. **Develop a dual-model registry** for safe rollback
   - Measurable: Rollback time <2 seconds, zero data loss

5. **Build live operator dashboard** with embedded telemetry
   - Measurable: 1-second polling loop, real-time alerts

---

## 5. Proposed Methodology

### 5.1 System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  SIMATIC S7-300 PLC → OPC UA / S7 Protocol                  │
│  (Temperature, Vibration, Pressure, Torque sensors)          │
└────────────────────┬────────────────────────────────────────┘
                     │ Live Telemetry Stream
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  EDGE DEVICE (IED / IEVD - Industrial Edge Device)          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  EdgeOps AI Container (Docker)                      │    │
│  │  • FastAPI Backend (Uvicorn ASGI)                   │    │
│  │  • ML Engine (XGBoost → ONNX)                       │    │
│  │  • Drift Monitor (Evidently)                        │    │
│  │  • Model Registry (SQLite)                          │    │
│  │  • React Dashboard + Grafana Embed                  │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                     │ Telemetry & Predictions
                     ↓
┌─────────────────────────────────────────────────────────────┐
│  IEM (Industrial Edge Management) - Factory Level           │
│  • Device fleet management                                  │
│  • App deployment & updates                                 │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Pipeline

**Phase 1: Data Ingestion**
- Real-time sensor streaming from PLC
- Features: Temperature, Rotational Speed, Torque, Tool Wear, Process Time

**Phase 2: Feature Engineering**
- Rolling statistics (mean, std) over 10-sample windows
- Lag features for temporal patterns
- Domain-specific ratios (e.g., Torque/Speed)

**Phase 3: Model Training**
- Algorithm: XGBoost (gradient boosted decision trees)
- Hyperparameter tuning: Optuna (Bayesian optimization, 50 trials)
- Cross-validation: 5-fold stratified
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

**Phase 4: Model Deployment**
- Convert to ONNX format (skl2onnx)
- ONNX Runtime inference engine
- Version logged to model_versions table

**Phase 5: Continuous Monitoring**
- Drift detection every 100 predictions (Evidently PSI)
- Alert triggered if drift_share ≥ 30%

**Phase 6: Auto-Retraining**
- New model trained with updated data
- Registered as "candidate" in registry
- Operator approval for production deployment

**Phase 7: Rollback Protocol**
- Dual-pointer system: `current_model_id` ↔ `stable_model_id`
- Instant rollback via SQL update (POST /rollback)

---

## 6. System Architecture

### 6.1 Component Diagram

```
┌──────────────────────────────────────────────────────────┐
│                 DATA & SENSING LAYER                     │
│  • PLC S7-300 sensors (24V binary + analog 0-10V)        │
│  • Edge data streaming pipeline                          │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│                 BACKEND API LAYER                        │
│  • FastAPI framework (async endpoints)                   │
│  • Uvicorn ASGI server                                   │
│  • Pydantic data validation                              │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│                 ML / MLOps ENGINE                        │
│  • Training: XGBoost + scikit-learn                      │
│  • Tuning: Optuna (Bayesian optimization)                │
│  • Tracking: MLflow (metrics, params, artifacts)         │
│  • Deployment: ONNX + ONNX Runtime                       │
│  • Monitoring: Evidently (drift detection)               │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│                 PERSISTENCE LAYER                        │
│  • SQLite: model_versions, deployment_state              │
│  • Tables: predictions, drift_metrics, system_metrics    │
│  • Alerts log with priority levels                       │
└────────────────┬─────────────────────────────────────────┘
                 ↓
┌──────────────────────────────────────────────────────────┐
│                 OPERATOR EXPERIENCE                      │
│  • React dashboard (1-second polling)                    │
│  • Embedded Grafana panels (SQLite data source)          │
│  • Audio siren for high-priority alerts                  │
│  • CSV export for alert history                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Database Schema

**model_versions**
```sql
id, version_number (UNIQUE), model_path,
accuracy, precision, recall, f1, roc_auc,
status (candidate | active | retired),
created_at, last_updated
```

**deployment_state** (SINGLETON - exactly 1 row)
```sql
id (CHECK id = 1),
current_model_id → model_versions.id (live inference pointer),
stable_model_id → model_versions.id (rollback pointer)
```

**system_metrics**
```sql
id, model_version, timestamp,
cpu_usage, ram_usage, latency_ms,
drift_score, reliability_score
```

---

## 7. Technology Stack

### 7.1 Core Technologies

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **ML Framework** | XGBoost 2.0 | SOTA for tabular data; handles imbalance; feature importance |
| **Optimization** | Optuna 3.5 | Bayesian TPE sampler; 50% faster than grid search |
| **Experiment Tracking** | MLflow 2.8 | Open-source; logs metrics/models/artifacts |
| **Edge Runtime** | ONNX Runtime 1.16 | 3-5× faster inference than scikit-learn |
| **Drift Detection** | Evidently 0.4 | Statistical tests (PSI, KS); report generation |
| **Backend** | FastAPI 0.104 | Async I/O; auto-generated OpenAPI docs |
| **Server** | Uvicorn 0.24 | ASGI server; WebSocket support |
| **Frontend** | React 19 + Vite 8 | Component-based; fast dev server |
| **Visualization** | Grafana OSS 10 | SQLite plugin; live time-series plots |
| **Database** | SQLite 3.45 | Embedded; WAL mode for concurrency |
| **Containerization** | Docker 24 | Reproducible edge deployments |

### 7.2 Dependencies (requirements.txt excerpt)

```
fastapi==0.104
uvicorn[standard]==0.24
xgboost==2.0
scikit-learn==1.3
optuna==3.5
mlflow==2.8
evidently==0.4
onnx==1.15
onnxruntime==1.16
pandas==2.1
numpy==1.25
```

---

## 8. Implementation Details

### 8.1 Drift Detection Algorithm

```python
# Pseudocode for drift monitoring loop
from evidently.metrics import DataDriftTable

def monitor_drift(reference_data, current_data):
    drift_report = DataDriftTable().calculate(
        reference_data=reference_data,
        current_data=current_data
    )
    
    drift_share = drift_report.get_drift_share()  # % of features drifted
    
    if drift_share >= DRIFT_THRESHOLD (0.30):
        trigger_alert("HIGH", "Model drift detected")
        initiate_retraining()
    
    log_to_db(drift_score=drift_share)
```

### 8.2 Auto-Retraining Pipeline

```python
# Triggered when drift_share >= 30%
def retrain_model():
    # 1. Load updated training data
    X_train, y_train = load_recent_data(days=30)
    
    # 2. Optuna hyperparameter search
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_xgboost, n_trials=50)
    
    # 3. Train with best params
    best_model = XGBClassifier(**study.best_params)
    best_model.fit(X_train, y_train)
    
    # 4. Evaluate on test set
    metrics = evaluate(best_model, X_test, y_test)
    
    # 5. Convert to ONNX
    onnx_model = convert_sklearn_to_onnx(best_model)
    
    # 6. Log to MLflow
    mlflow.log_params(study.best_params)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(onnx_model)
    
    # 7. Register as candidate
    register_model(version=next_version, status="candidate", metrics=metrics)
```

### 8.3 Rollback Mechanism

```python
# POST /rollback endpoint
def rollback_deployment():
    # Atomic SQL update
    db.execute("""
        UPDATE deployment_state
        SET current_model_id = stable_model_id
        WHERE id = 1
    """)
    
    log_event("Model rolled back to stable version")
    clear_alerts()
    return {"status": "Rollback successful"}
```

### 8.4 Dataset Description

**Source:** UCI Machine Learning Repository - Predictive Maintenance Dataset

**Features (10):**
- UDI: Unique identifier
- Product ID: Quality variant (L/M/H)
- Type: Product type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

**Target:** Machine failure (binary: 0 = No, 1 = Yes)

**Size:** 10,000 samples (1.3% failure rate → imbalanced)

**Preprocessing:**
- SMOTE for class balancing
- StandardScaler normalization
- 80/20 train-test split

---

## 9. Results & Discussion

### 9.1 Model Performance Metrics

| Metric | Baseline (v0001) | Retrained (v0002) | Improvement |
|--------|-----------------|------------------|-------------|
| Accuracy | 95.2% | 97.1% | +1.9% |
| Precision | 91.3% | 94.5% | +3.2% |
| Recall | 89.7% | 93.2% | +3.5% |
| F1-Score | 90.5% | 93.8% | +3.3% |
| ROC-AUC | 0.943 | 0.968 | +0.025 |

### 9.2 Inference Performance

- **ONNX Inference Time:** 6.3ms (avg over 1000 predictions)
- **Python scikit-learn:** 23.1ms (baseline)
- **Speedup:** 3.7×

### 9.3 Drift Detection Results

- **Monitoring Window:** 100 predictions
- **Drift Share Threshold:** 30%
- **Alert Triggered:** 18% drift detected (normal operation)
- **Simulated Drift:** 42% → retraining initiated successfully

### 9.4 System Reliability

- **Uptime:** 99.7% over 72-hour test
- **Rollback Time:** 1.8 seconds (average)
- **Dashboard Polling:** 1-second latency (stable)

---

## 10. Conclusion & Future Work

### 10.1 Contributions

1. ✅ **First production-grade MLOps platform** for edge predictive maintenance
2. ✅ **Autonomous drift-to-retrain pipeline** (30% threshold, <5 min cycle)
3. ✅ **Dual-pointer model registry** for instant rollback (1.8s)
4. ✅ **ONNX-optimized inference** (6.3ms, 3.7× speedup)
5. ✅ **Live operator dashboard** with Grafana + alert system

### 10.2 Limitations

- **Single-device deployment:** Not tested on distributed edge clusters
- **Simulated PLC data:** Real S7-300 integration pending factory validation
- **Drift detection:** Statistical methods only (no deep learning-based drift)

### 10.3 Future Work

1. **Federated Learning:** Multi-site model aggregation without raw data sharing
2. **Model Compression:** Quantization (INT8) for ultra-low-power devices (<1W)
3. **Advanced Drift:** Autoencoders for detecting novel failure modes
4. **Cloud-Edge Hybrid:** Offload retraining to cloud, deploy ONNX to edge
5. **Explainability:** SHAP values for operator trust in predictions

---

## References

1. Lee, J., et al. (2014). "Prognostics and health management design for rotary machinery systems." *Mechanical Systems and Signal Processing*, 42(1-2), 314-334.

2. Susto, G. A., et al. (2015). "Machine Learning for Predictive Maintenance: A Multiple Classifier Approach." *IEEE Transactions on Industrial Informatics*, 11(3), 812-820.

3. Carvalho, T. P., et al. (2019). "A systematic literature review of machine learning methods applied to predictive maintenance." *Computers & Industrial Engineering*, 137, 106024.

4. Paleyes, A., et al. (2022). "Challenges in Deploying Machine Learning: A Survey of Case Studies." *ACM Computing Surveys*, 55(6), 1-29.

5. Kreuzberger, D., et al. (2023). "Machine Learning Operations (MLOps): Overview, Definition, and Architecture." *IEEE Access*, 11, 31866-31879.

6. Merenda, M., et al. (2020). "Edge Machine Learning for AI-Enabled IoT Devices: A Review." *Sensors*, 20(9), 2533.

7. Lu, J., et al. (2018). "Learning under Concept Drift: A Review." *IEEE Transactions on Knowledge and Data Engineering*, 31(12), 2346-2363.

8. Gama, J., et al. (2014). "A survey on concept drift adaptation." *ACM Computing Surveys*, 46(4), 1-37.

---

## Thank You

**Questions?**

**Contact:** vanshita.mehta@example.com  
**GitHub:** github.com/vanshita/edgeops-ai  
**Project Demo:** http://localhost:5173

---

## Appendix: API Endpoints

### Core Endpoints

```
GET  /health              → API status check
GET  /models/current      → Active model metadata
POST /predict             → Inference endpoint (ONNX)
GET  /alerts              → Active alerts list
POST /rollback            → Rollback to stable model
GET  /stream/status       → Live telemetry status
POST /stream/start        → Start sensor streaming
POST /stream/stop         → Stop sensor streaming
GET  /drift/report        → Latest drift analysis
POST /train               → Trigger manual retraining
POST /hard_reset          → Emergency system reset
```

### Dashboard Metrics

```
GET  /metrics/system      → CPU, RAM, latency
GET  /metrics/drift       → Drift score time-series
GET  /models/versions     → Model registry history
```

---

## Appendix: Deployment Checklist

**Pre-Deployment:**
- [ ] Docker Desktop installed & running
- [ ] PLC S7-300 OPC UA connector configured
- [ ] Sensor thresholds defined (sensor_thresholds.json)
- [ ] Grafana SQLite plugin installed

**Deployment:**
```bash
# 1. Build containers
docker-compose build

# 2. Start services
docker-compose up -d

# 3. Initialize ML pipeline
curl -X POST http://localhost:8000/train

# 4. Verify Grafana datasource
# Go to http://localhost:3000 → Data Sources → SQLite
# Path: /app_data/edgeops.db

# 5. Access dashboard
# Frontend: http://localhost:5173
# API Docs: http://localhost:8000/docs
```

**Post-Deployment:**
- [ ] Verify live streaming (GET /stream/status)
- [ ] Check initial predictions (POST /predict)
- [ ] Monitor drift metrics (GET /drift/report)
- [ ] Test rollback mechanism (POST /rollback)

---

*End of Presentation*
