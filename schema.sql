-- schema.sql
-- Edge AI Operations Registry Blueprint

-- 1. TRACKS COMPILED ML MODELS AND VALIDATION SCORES
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version_number INTEGER NOT NULL UNIQUE,
    model_path TEXT NOT NULL,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    roc_auc REAL,
    status TEXT DEFAULT 'candidate',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. ENSURES AN INFINITE SINGLETON ACTIVE EDGE ARCHITECTURE
CREATE TABLE IF NOT EXISTS deployment_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    current_model_id INTEGER,
    stable_model_id INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (current_model_id) REFERENCES model_versions(id),
    FOREIGN KEY (stable_model_id) REFERENCES model_versions(id)
);

-- 3. TRACKS LIVE INDUSTRIAL DATA STREAMING AND COMPUTE HEALTH
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version INTEGER,               -- Links metrics to the model executing at that second
    cpu_usage REAL NOT NULL,             -- Live hardware processor load (%)
    ram_usage REAL NOT NULL,             -- Live edge device memory consumption (%)
    latency_ms REAL NOT NULL,            -- Execution delay of the endpoint (milliseconds)
    drift_score REAL DEFAULT 0.0,        -- Current PSI or KS drift calculation status
    reliability_score REAL DEFAULT 100.0,-- Combined certainty matrix calculation (%)
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (model_version) REFERENCES model_versions(version_number)
);