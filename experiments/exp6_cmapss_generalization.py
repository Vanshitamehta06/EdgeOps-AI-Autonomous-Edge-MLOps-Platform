"""
Experiment 6: Cross-Domain Generalization on NASA C-MAPSS Turbofan Engine Dataset.
Validates the EdgeOps predictive maintenance pipeline on NASA C-MAPSS FD001 run-to-failure telemetry.
Measures generalization performance, GroupKFold cross-validation, degradation curves,
and ONNX edge compilation latency.
"""

import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier
import onnxruntime as ort
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CMAPSS_FILE = BASE_DIR / "data_cmapss_train_FD001.txt"

# Column definitions for NASA C-MAPSS
INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# Sensors with non-zero variance under constant sea-level condition (FD001 standard subset)
USEFUL_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9", "s_11", "s_12", "s_13",
    "s_14", "s_15", "s_17", "s_20", "s_21"
]
FEATURE_COLS = SETTING_COLS + USEFUL_SENSORS


def load_and_preprocess_cmapss(file_path: Path, failure_window: int = 30):
    """Load C-MAPSS FD001, compute RUL, and define binary failure window."""
    df = pd.read_csv(file_path, sep=r"\s+", header=None, names=ALL_COLS)
    
    # Calculate Max Cycles per engine
    max_cycles = df.groupby("unit_nr")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_nr", "max_cycles"]
    df = df.merge(max_cycles, on="unit_nr", how="left")
    
    # RUL = max_cycles - time_cycles
    df["RUL"] = df["max_cycles"] - df["time_cycles"]
    
    # Binary target: 1 if RUL <= failure_window else 0
    df["failure_imminent"] = (df["RUL"] <= failure_window).astype(int)
    
    return df


def run_cmapss_benchmark():
    print("=" * 70)
    print("NASA C-MAPSS Turbofan Engine Predictive Maintenance Benchmark")
    print("=" * 70)

    df = load_and_preprocess_cmapss(CMAPSS_FILE, failure_window=30)
    print(f"Loaded {len(df):,} records across {df['unit_nr'].nunique()} turbofan engines.")
    class_counts = df["failure_imminent"].value_counts().to_dict()
    print(f"Target distribution -> Healthy (0): {class_counts.get(0, 0)} | Imminent Failure (1): {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.2f}%)")

    X = df[FEATURE_COLS].values
    y = df["failure_imminent"].values
    groups = df["unit_nr"].values

    # 5-Fold GroupKFold Cross-Validation (Ensures no engine leaks between train and test)
    gkf = GroupKFold(n_splits=5)
    
    fold_metrics = []
    y_test_all = []
    y_pred_all = []
    y_prob_all = []

    print("\nExecuting 5-Fold Engine-Grouped Cross-Validation...")
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Calculate scale_pos_weight
        spw = (len(y_train) - sum(y_train)) / sum(y_train)

        clf = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42 + fold
        )

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ROC_AUC": roc_auc,
            "PR_AUC": pr_auc
        })

        y_test_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_prob_all.extend(y_prob)

        print(f"  Fold {fold} -> Acc: {acc*100:.2f}% | F1: {f1:.4f} | Recall: {rec*100:.2f}% | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")

    metrics_df = pd.DataFrame(fold_metrics)
    mean_row = metrics_df.mean(numeric_only=True)
    std_row = metrics_df.std(numeric_only=True)

    print("\n--- 5-Fold Cross-Validation Summary ---")
    print(f"Accuracy:  {mean_row['Accuracy']*100:.2f}% +/- {std_row['Accuracy']*100:.2f}%")
    print(f"Precision: {mean_row['Precision']*100:.2f}% +/- {std_row['Precision']*100:.2f}%")
    print(f"Recall:    {mean_row['Recall']*100:.2f}% +/- {std_row['Recall']*100:.2f}%")
    print(f"F1-Score:  {mean_row['F1']:.4f} +/- {std_row['F1']:.4f}")
    print(f"ROC-AUC:   {mean_row['ROC_AUC']:.4f} +/- {std_row['ROC_AUC']:.4f}")
    print(f"PR-AUC:    {mean_row['PR_AUC']:.4f} +/- {std_row['PR_AUC']:.4f}")

    # Train final model on full dataset and export to ONNX
    print("\nCompiling Full C-MAPSS Edge Model to ONNX Runtime...")
    final_spw = (len(y) - sum(y)) / sum(y)
    full_model = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=final_spw,
        eval_metric="logloss", random_state=42
    )
    full_model.fit(X, y)

    # Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, len(FEATURE_COLS)]))]
    onnx_model = onnxmltools.convert_xgboost(full_model, initial_types=initial_type)
    onnx_path = RESULTS_DIR / "cmapss_edge_model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    # Benchmark Latency: Native XGBoost vs ONNX Runtime
    sample_feat = X[:1000].astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # ONNX benchmark
    t0 = time.perf_counter()
    for row in sample_feat:
        _ = session.run(None, {input_name: row.reshape(1, -1)})
    onnx_lat_us = ((time.perf_counter() - t0) / 1000.0) * 1e6

    # Native benchmark
    t0 = time.perf_counter()
    for row in sample_feat:
        _ = full_model.predict_proba(row.reshape(1, -1))
    native_lat_us = ((time.perf_counter() - t0) / 1000.0) * 1e6

    print(f"Native XGBoost Single-Sample Latency: {native_lat_us:.1f} µs ({native_lat_us/1000.0:.3f} ms)")
    print(f"ONNX Runtime Single-Sample Latency:    {onnx_lat_us:.1f} µs ({onnx_lat_us/1000.0:.3f} ms)")
    print(f"ONNX Latency Speedup:                 {native_lat_us / onnx_lat_us:.2f}x")

    # Generate Visualization Figure
    print("\nGenerating NASA C-MAPSS Degradation & Performance Charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Sensor Degradation Run-to-Failure for Engine #1 and #2
    ax1 = axes[0]
    for unit_id, color in [(1, "#2563eb"), (2, "#dc2626")]:
        unit_df = df[df["unit_nr"] == unit_id]
        ax1.plot(unit_df["time_cycles"], unit_df["s_11"], label=f"Engine {unit_id} (HPC Outlet Pres)", color=color, lw=2)
        # Mark failure point
        max_cyc = unit_df["time_cycles"].max()
        ax1.axvline(max_cyc, color=color, linestyle="--", alpha=0.6)
        ax1.axvspan(max_cyc - 30, max_cyc, color=color, alpha=0.1, label=f"Engine {unit_id} Warning Window" if unit_id == 1 else "")

    ax1.set_title("Turbofan Sensor Degradation (Run-to-Failure)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Operational Cycles")
    ax1.set_ylabel("Sensor 11 (Static Pressure, psia)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=9)

    # Plot 2: Confusion Matrix
    ax2 = axes[1]
    cm = confusion_matrix(y_test_all, y_pred_all)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Failure"])
    disp.plot(cmap="Blues", ax=ax2, values_format="d", colorbar=False)
    ax2.set_title(f"GroupKFold Confusion Matrix (N={len(y_test_all):,})", fontsize=12, fontweight="bold")

    # Plot 3: Metrics comparison bar chart
    ax3 = axes[2]
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    metric_values = [mean_row[m] for m in metric_names]
    metric_errors = [std_row[m] for m in metric_names]
    bars = ax3.bar(metric_names, metric_values, yerr=metric_errors, capsize=5, color="#10b981", edgecolor="#047857", alpha=0.85)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Cross-Validated Generalization Metrics", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 0.03, f"{h:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    chart_path = RESULTS_DIR / "cmapss_generalization_degradation.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved visualization plot to {chart_path}")

    # Save Metrics CSV
    summary_dict = {
        "Dataset": "NASA C-MAPSS FD001 Turbofan",
        "Total Records": len(df),
        "Total Engines": df["unit_nr"].nunique(),
        "CV Scheme": "5-Fold GroupKFold (Unseen Engines)",
        "Mean Accuracy": round(mean_row["Accuracy"], 4),
        "Accuracy Std": round(std_row["Accuracy"], 4),
        "Mean Precision": round(mean_row["Precision"], 4),
        "Precision Std": round(std_row["Precision"], 4),
        "Mean Recall": round(mean_row["Recall"], 4),
        "Recall Std": round(std_row["Recall"], 4),
        "Mean F1": round(mean_row["F1"], 4),
        "F1 Std": round(std_row["F1"], 4),
        "Mean ROC_AUC": round(mean_row["ROC_AUC"], 4),
        "ROC_AUC Std": round(std_row["ROC_AUC"], 4),
        "Mean PR_AUC": round(mean_row["PR_AUC"], 4),
        "PR_AUC Std": round(std_row["PR_AUC"], 4),
        "ONNX Latency (us)": round(onnx_lat_us, 2),
        "Native Latency (us)": round(native_lat_us, 2),
        "ONNX Speedup": round(native_lat_us / onnx_lat_us, 2)
    }
    summary_df = pd.DataFrame([summary_dict])
    csv_path = RESULTS_DIR / "cmapss_generalization_metrics.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved generalization metrics to {csv_path}")

    print("\n=== P5 NASA C-MAPSS Benchmark Complete Successfully ===")


if __name__ == "__main__":
    run_cmapss_benchmark()
