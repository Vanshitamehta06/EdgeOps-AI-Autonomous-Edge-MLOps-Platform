"""
experiments/exp4_edge_benchmark.py

P1d: Edge Hardware Inference & Resource Benchmark.
Compares Native Scikit-Learn/XGBoost vs. ONNX Runtime on CPU across:
1. Latency Distribution: Mean, Median (p50), p90, p95, p99 over 1,000 inference runs.
2. Throughput: Inferences per second (FPS/Hz).
3. Artifact Binary Footprint: Disk size (.pkl vs .onnx).
4. System Hardware Resource Utilization: CPU load (%) and Memory Consumption (RAM RSS MB) via psutil.

Outputs saved to results/.
"""

import sys
import os
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import psutil
import joblib
import onnxruntime as ort
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
import ml_pipeline

plt.switch_backend('Agg')

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_ITERATIONS = 1000

def get_process_memory_mb():
    """Returns current process RAM RSS in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_experiment():
    print(f"=== P1d: Edge Latency & Resource Benchmark ({N_ITERATIONS} Runs) ===")

    # 1. Load active production model bundle
    current_deployment = Path(config.MODELS_DIR / "version_0003")
    if not current_deployment.exists():
        versions = sorted(list(config.MODELS_DIR.glob("version_*")))
        current_deployment = versions[-1] if versions else None
        
    if not current_deployment:
        raise FileNotFoundError("No trained model bundle found in models/ directory.")

    print(f"Evaluating model bundle at: {current_deployment}")
    
    pkl_path = current_deployment / "model.pkl"
    onnx_path = current_deployment / "model.onnx"
    
    pkl_size_kb = os.path.getsize(pkl_path) / 1024.0
    onnx_size_kb = os.path.getsize(onnx_path) / 1024.0
    print(f"Model File Sizes -> Native Pickle: {pkl_size_kb:.2f} KB | ONNX Binary: {onnx_size_kb:.2f} KB (Compression: {(1 - onnx_size_kb/pkl_size_kb)*100:.1f}%)")

    # 2. Load Native XGBoost Model
    mem_before_xgb = get_process_memory_mb()
    xgb_model = joblib.load(pkl_path)
    mem_after_xgb = get_process_memory_mb()

    # 3. Load ONNX Runtime Session
    mem_before_onnx = get_process_memory_mb()
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    onnx_session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = onnx_session.get_inputs()[0].name
    mem_after_onnx = get_process_memory_mb()

    # Create realistic single-sample payload
    sample_payload = pd.DataFrame([{
        "Type": 1,
        "Air temperature K": 300.0,
        "Process temperature K": 310.0,
        "Rotational speed rpm": 1500.0,
        "Torque Nm": 40.0,
        "Tool wear min": 100.0
    }])
    sample_numpy = sample_payload.astype(np.float32).values

    # Warmup runs (JIT & caching warmup)
    for _ in range(50):
        _ = xgb_model.predict(sample_payload)
        _ = onnx_session.run(None, {input_name: sample_numpy})

    # 4. Benchmark Native XGBoost
    print("\nBenchmarking Native XGBoost CPU Latency (1,000 single predictions)...")
    xgb_latencies = []
    cpu_samples_xgb = []
    
    for i in range(N_ITERATIONS):
        t0 = time.perf_counter()
        _ = xgb_model.predict(sample_payload)
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        xgb_latencies.append(t_elapsed)
        if i % 100 == 0:
            cpu_samples_xgb.append(psutil.cpu_percent(interval=None))

    # 5. Benchmark ONNX Runtime
    print("Benchmarking ONNX Runtime CPU Latency (1,000 single predictions)...")
    onnx_latencies = []
    cpu_samples_onnx = []
    
    for i in range(N_ITERATIONS):
        t0 = time.perf_counter()
        _ = onnx_session.run(None, {input_name: sample_numpy})
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        onnx_latencies.append(t_elapsed)
        if i % 100 == 0:
            cpu_samples_onnx.append(psutil.cpu_percent(interval=None))

    # Compute Statistics
    xgb_lat = np.array(xgb_latencies)
    onnx_lat = np.array(onnx_latencies)

    metrics = {
        "Metric": [
            "Mean Latency (ms)",
            "Median / p50 Latency (ms)",
            "p90 Latency (ms)",
            "p95 Latency (ms)",
            "p99 Latency (ms)",
            "Min Latency (ms)",
            "Max Latency (ms)",
            "Throughput (Inferences/sec)",
            "Artifact Size (KB)",
            "Memory Loaded (MB)"
        ],
        "Native_XGBoost": [
            round(float(np.mean(xgb_lat)), 3),
            round(float(np.percentile(xgb_lat, 50)), 3),
            round(float(np.percentile(xgb_lat, 90)), 3),
            round(float(np.percentile(xgb_lat, 95)), 3),
            round(float(np.percentile(xgb_lat, 99)), 3),
            round(float(np.min(xgb_lat)), 3),
            round(float(np.max(xgb_lat)), 3),
            round(float(1000.0 / np.mean(xgb_lat)), 1),
            round(pkl_size_kb, 2),
            round(max(mem_after_xgb - mem_before_xgb, 0.1), 2)
        ],
        "ONNX_Runtime_CPU": [
            round(float(np.mean(onnx_lat)), 3),
            round(float(np.percentile(onnx_lat, 50)), 3),
            round(float(np.percentile(onnx_lat, 90)), 3),
            round(float(np.percentile(onnx_lat, 95)), 3),
            round(float(np.percentile(onnx_lat, 99)), 3),
            round(float(np.min(onnx_lat)), 3),
            round(float(np.max(onnx_lat)), 3),
            round(float(1000.0 / np.mean(onnx_lat)), 1),
            round(onnx_size_kb, 2),
            round(max(mem_after_onnx - mem_before_onnx, 0.1), 2)
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    p50_speedup = metrics["Native_XGBoost"][1] / metrics["ONNX_Runtime_CPU"][1]
    p95_speedup = metrics["Native_XGBoost"][3] / metrics["ONNX_Runtime_CPU"][3]
    
    print("\n--- Benchmark Summary Table ---")
    print(df_metrics.to_string(index=False))
    print(f"\n=> ONNX Runtime Speedup: {p50_speedup:.2f}x (p50) | {p95_speedup:.2f}x (p95)")

    csv_path = RESULTS_DIR / "edge_benchmark_metrics.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to {csv_path}")

    # Plot Latency Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Boxplot
    axes[0].boxplot([xgb_lat, onnx_lat], tick_labels=["Native XGBoost", "ONNX Runtime CPU"],
                    patch_artist=True, boxprops=dict(facecolor="#93c5fd"), medianprops=dict(color="red", linewidth=2))
    axes[0].set_ylabel("Inference Latency (milliseconds)", fontsize=11)
    axes[0].set_title("Latency Distribution (p50 / Interquartile)", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Histogram / KDE curves
    axes[1].hist(xgb_lat, bins=40, alpha=0.6, color="#ef4444", label=f"Native XGBoost (p50: {metrics['Native_XGBoost'][1]} ms)")
    axes[1].hist(onnx_lat, bins=40, alpha=0.7, color="#22c55e", label=f"ONNX Runtime (p50: {metrics['ONNX_Runtime_CPU'][1]} ms)")
    axes[1].set_xlabel("Latency (milliseconds)", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Edge Inference Latency Histogram (1,000 Runs)", fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "edge_latency_distribution.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved latency distribution plot to {plot_path}")

    print("\n=== P1d Complete Successfully ===")

if __name__ == "__main__":
    run_experiment()
