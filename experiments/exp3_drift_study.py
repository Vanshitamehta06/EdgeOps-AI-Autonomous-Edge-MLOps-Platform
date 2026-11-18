"""
experiments/exp3_drift_study.py

P1c: Controlled Data Drift and Self-Healing Retraining Study.
1. Injects controlled physical drift (0.5 to 3.0 standard deviations on Torque and Temperature).
2. Measures Detection Delay (minimum sample size needed to detect drift).
3. Measures False Alarm Rate on a clean industrial stream across threshold settings (10% to 50%).
4. Evaluates performance (Accuracy, F1, ROC-AUC) comparing a Static Model vs. Self-Healing Loop.
5. Performs a threshold sweep (10% - 50%) to empirically justify the 30% drift threshold.

Outputs saved to results/.
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
import ml_pipeline
import monitor

plt.switch_backend('Agg')

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def run_experiment():
    print(f"=== P1c: Controlled Drift & Self-Healing Study ===")
    
    # 1. Load baseline data
    raw_df = pd.read_csv(config.DATASET_PATH)
    ref_df = raw_df.drop(columns=config.COLUMNS_TO_DROP, errors='ignore')
    if config.TARGET_COLUMN in ref_df.columns:
        ref_features = ref_df.drop(columns=[config.TARGET_COLUMN])
        ref_target = ref_df[config.TARGET_COLUMN]
    else:
        ref_features = ref_df
        ref_target = None
        
    # Map Type to numerical
    if 'Type' in ref_features.columns and ref_features['Type'].dtype == 'object':
        ref_features['Type'] = ref_features['Type'].map({'L': 0, 'M': 1, 'H': 2}).fillna(0)
        
    num_features = len(ref_features.columns)
    print(f"Reference features ({num_features}): {list(ref_features.columns)}")

    # Consistent feature naming across all splits and batches
    X, y, _ = ml_pipeline.preprocess_data(raw_df)
    ref_features = X.copy()
    X_train, X_test, y_train, y_test = ml_pipeline.split_data(X, y)
    
    scale_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    static_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    static_model.fit(X_train, y_train)
    
    baseline_f1 = f1_score(y_test, static_model.predict(X_test), zero_division=0)
    print(f"Static Model clean test F1: {baseline_f1:.4f}")

    # =========================================================================
    # PART 1: False Alarm Rate on Clean Stream
    # =========================================================================
    print("\n--- Part 1: Clean Stream False Alarm Rate across Thresholds ---")
    threshold_candidates = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    n_clean_batches = 30
    batch_size = 200
    
    clean_alarm_counts = {th: 0 for th in threshold_candidates}
    drift_shares_clean = []
    
    for i in range(n_clean_batches):
        clean_batch = ref_features.sample(n=batch_size, replace=True, random_state=RANDOM_STATE + i)
        
        rep = monitor.analyze_data_drift(ref_features, clean_batch)
        share = float(rep["drift_share"])
        drift_shares_clean.append(share)
        
        for th in threshold_candidates:
            if share >= th:
                clean_alarm_counts[th] += 1

    clean_results = []
    for th in threshold_candidates:
        far = (clean_alarm_counts[th] / n_clean_batches) * 100.0
        clean_results.append({
            "Threshold": th,
            "Threshold_Pct": f"{int(th*100)}%",
            "False_Alarms": clean_alarm_counts[th],
            "Total_Clean_Batches": n_clean_batches,
            "False_Alarm_Rate_Pct": round(far, 2)
        })
        print(f"Threshold {int(th*100)}%: False Alarm Rate = {far:.1f}% ({clean_alarm_counts[th]}/{n_clean_batches})")

    clean_df = pd.DataFrame(clean_results)
    clean_df.to_csv(RESULTS_DIR / "drift_clean_stream_false_alarms.csv", index=False)

    # =========================================================================
    # PART 2: Controlled Drift Injection & Static vs. Retrained Comparison
    # =========================================================================
    print("\n--- Part 2: Controlled Drift Severity Sweep (0.5 to 3.0 SD) ---")
    severities = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    torque_col = [c for c in X.columns if "Torque" in c][0]
    temp_col = [c for c in X.columns if "Air temperature" in c][0]
    
    torque_std = float(X[torque_col].std())
    temp_std = float(X[temp_col].std())
    
    drift_eval_rows = []
    
    for sd in severities:
        print(f"\nEvaluating Drift Severity: +{sd} SD (Torque +{sd*torque_std:.1f} Nm, Temp +{sd*temp_std:.1f} K)...")
        
        # Create drifted holdout test set
        X_test_drifted = X_test.copy()
        X_test_drifted[torque_col] += sd * torque_std
        X_test_drifted[temp_col] += sd * temp_std
        
        # Measure Static Model performance on drifted data
        y_pred_static = static_model.predict(X_test_drifted)
        y_prob_static = static_model.predict_proba(X_test_drifted)[:, 1]
        
        static_f1 = f1_score(y_test, y_pred_static, zero_division=0)
        static_acc = accuracy_score(y_test, y_pred_static)
        static_roc = roc_auc_score(y_test, y_prob_static)
        
        # Test Drift Detection on this batch
        test_batch = ref_features.sample(n=200, replace=True, random_state=RANDOM_STATE).copy()
        test_batch[torque_col] += sd * torque_std
        test_batch[temp_col] += sd * temp_std
        
        rep = monitor.analyze_data_drift(ref_features, test_batch)
        drift_share = float(rep["drift_share"])
        is_detected_30 = drift_share >= 0.30
        
        # Simulate detection delay (sweep batch sizes: 30, 50, 75, 100, 150, 200, 300)
        detection_sample_size = None
        for sample_n in [30, 50, 75, 100, 150, 200, 300]:
            sub_batch = ref_features.sample(n=sample_n, replace=True, random_state=RANDOM_STATE).copy()
            sub_batch[torque_col] += sd * torque_std
            sub_batch[temp_col] += sd * temp_std
            sub_rep = monitor.analyze_data_drift(ref_features, sub_batch)
            if sub_rep["drift_share"] >= 0.30:
                detection_sample_size = sample_n
                break
        if detection_sample_size is None:
            detection_sample_size = ">300"
            
        # Simulate Self-Healing Loop: Retrain model on drifted distribution
        X_train_drifted = X_train.copy()
        X_train_drifted[torque_col] += sd * torque_std
        X_train_drifted[temp_col] += sd * temp_std
        
        retrained_model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        retrained_model.fit(X_train_drifted, y_train)
        
        y_pred_retrained = retrained_model.predict(X_test_drifted)
        y_prob_retrained = retrained_model.predict_proba(X_test_drifted)[:, 1]
        
        retrained_f1 = f1_score(y_test, y_pred_retrained, zero_division=0)
        retrained_acc = accuracy_score(y_test, y_pred_retrained)
        retrained_roc = roc_auc_score(y_test, y_prob_retrained)
        
        f1_recovery_delta = retrained_f1 - static_f1
        
        drift_eval_rows.append({
            "Drift_Severity_SD": sd,
            "Drift_Share": round(drift_share, 4),
            "Detected_at_30Pct": is_detected_30,
            "Min_Detection_Samples": str(detection_sample_size),
            "Static_F1": round(static_f1, 4),
            "Static_Accuracy": round(static_acc, 4),
            "Static_ROC_AUC": round(static_roc, 4),
            "Retrained_F1": round(retrained_f1, 4),
            "Retrained_Accuracy": round(retrained_acc, 4),
            "Retrained_ROC_AUC": round(retrained_roc, 4),
            "F1_Recovery_Gain": round(f1_recovery_delta, 4)
        })
        
        print(f" -> Drift Share: {drift_share*100:.1f}% | Detected at 30%: {is_detected_30} | Delay: {detection_sample_size} samples")
        print(f" -> Static F1: {static_f1:.4f} vs Retrained F1: {retrained_f1:.4f} (Recovery Gain: +{f1_recovery_delta:.4f})")

    drift_df = pd.DataFrame(drift_eval_rows)
    drift_df.to_csv(RESULTS_DIR / "drift_severity_comparison.csv", index=False)

    # =========================================================================
    # PART 3: Threshold Justification Plotting
    # =========================================================================
    print("\nGenerating comprehensive Drift Study plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. False Alarm Rate vs Threshold
    axes[0].plot(clean_df["Threshold"] * 100, clean_df["False_Alarm_Rate_Pct"], 'ro-', linewidth=2, markersize=8)
    axes[0].axvline(30, color='blue', linestyle='--', label='Selected 30% Threshold')
    axes[0].set_xlabel("Drift Share Threshold (%)", fontsize=11)
    axes[0].set_ylabel("False Alarm Rate (%) on Clean Stream", fontsize=11)
    axes[0].set_title("False Alarm Rate vs. Drift Threshold", fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 2. Static vs Retrained F1 Score
    axes[1].plot(drift_df["Drift_Severity_SD"], drift_df["Static_F1"], 's--', color='#ef4444', linewidth=2, label='Static Model (Degraded)')
    axes[1].plot(drift_df["Drift_Severity_SD"], drift_df["Retrained_F1"], 'o-', color='#22c55e', linewidth=2, label='Self-Healing Model (Adapted)')
    axes[1].set_xlabel("Injected Drift Severity (Standard Deviations)", fontsize=11)
    axes[1].set_ylabel("Holdout F1-Score", fontsize=11)
    axes[1].set_title("Performance: Static vs. Self-Healing Closed Loop", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # 3. Drift Share vs Severity
    axes[2].plot(drift_df["Drift_Severity_SD"], drift_df["Drift_Share"] * 100, 'b^-', linewidth=2, markersize=8)
    axes[2].axhline(30, color='red', linestyle='--', label='Trigger Threshold (30%)')
    axes[2].set_xlabel("Injected Drift Severity (SD)", fontsize=11)
    axes[2].set_ylabel("Share of Drifted Features (%)", fontsize=11)
    axes[2].set_title("Drift Share vs. Physical Degradation", fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plots_path = RESULTS_DIR / "drift_study_plots.png"
    plt.savefig(plots_path, dpi=200)
    plt.close()
    print(f"Saved drift study plots to {plots_path}")

    print("\n=== P1c Complete Successfully ===")

if __name__ == "__main__":
    run_experiment()
