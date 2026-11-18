"""
experiments/exp1_model_comparison.py

P1a: Comprehensive Model Comparison on AI4I 2020 Predictive Maintenance Dataset.
Compares XGBoost vs. Logistic Regression vs. Random Forest.
Uses 5-Fold Stratified Cross-Validation with Class Imbalance compensation.
Computes: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrices, and SHAP explanations.
Outputs saved to results/.
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
import ml_pipeline

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
import shap

# Configure matplotlib for headless environment
plt.switch_backend('Agg')

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5

def run_experiment():
    print(f"=== P1a: Model Comparison on {config.DATASET_PATH.name} ===")
    
    # 1. Ingest Data
    raw_df = pd.read_csv(config.DATASET_PATH)
    total_records = len(raw_df)
    failure_count = int(raw_df[config.TARGET_COLUMN].sum())
    failure_rate = (failure_count / total_records) * 100.0
    print(f"Total Records: {total_records} | Failures: {failure_count} ({failure_rate:.2f}%)")
    
    X, y, encoder = ml_pipeline.preprocess_data(raw_df)
    feature_names = list(X.columns)
    print(f"Features ({len(feature_names)}): {feature_names}")
    
    # Calculate global negative/positive ratio
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_weight = float(neg_count / max(pos_count, 1))
    print(f"Class ratio: {neg_count}:{pos_count} (scale_pos_weight = {scale_weight:.2f})")
    
    # Define models with class imbalance mitigation
    models = {
        "Logistic_Regression": LogisticRegression(
            class_weight="balanced", 
            max_iter=1000, 
            random_state=RANDOM_STATE
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, 
            class_weight="balanced_subsample", 
            random_state=RANDOM_STATE, 
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
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
    }
    
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    metrics_summary = {}
    aggregated_cms = {name: np.zeros((2, 2), dtype=int) for name in models}
    
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name} across {N_SPLITS}-Fold Stratified CV...")
        fold_metrics = {
            "accuracy": [], "precision": [], "recall": [], 
            "f1": [], "roc_auc": [], "pr_auc": []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Logistic Regression requires feature standardization
            if model_name == "Logistic_Regression":
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_val = scaler.transform(X_val)
                
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            roc = roc_auc_score(y_val, y_proba)
            pr_auc = average_precision_score(y_val, y_proba)
            
            fold_metrics["accuracy"].append(acc)
            fold_metrics["precision"].append(prec)
            fold_metrics["recall"].append(rec)
            fold_metrics["f1"].append(f1)
            fold_metrics["roc_auc"].append(roc)
            fold_metrics["pr_auc"].append(pr_auc)
            
            aggregated_cms[model_name] += confusion_matrix(y_val, y_pred)
            
        # Compute mean and standard deviation
        model_result = {}
        for m, vals in fold_metrics.items():
            model_result[f"{m}_mean"] = float(np.mean(vals))
            model_result[f"{m}_std"] = float(np.std(vals))
            
        metrics_summary[model_name] = model_result
        print(f" -> Accuracy: {model_result['accuracy_mean']:.4f} ± {model_result['accuracy_std']:.4f}")
        print(f" -> F1 Score: {model_result['f1_mean']:.4f} ± {model_result['f1_std']:.4f}")
        print(f" -> Precision: {model_result['precision_mean']:.4f} ± {model_result['precision_std']:.4f}")
        print(f" -> Recall:    {model_result['recall_mean']:.4f} ± {model_result['recall_std']:.4f}")
        print(f" -> ROC-AUC:   {model_result['roc_auc_mean']:.4f} ± {model_result['roc_auc_std']:.4f}")
        print(f" -> PR-AUC:    {model_result['pr_auc_mean']:.4f} ± {model_result['pr_auc_std']:.4f}")
        
    # Save CSV and JSON
    summary_df = pd.DataFrame(metrics_summary).T
    csv_path = RESULTS_DIR / "model_comparison_metrics.csv"
    json_path = RESULTS_DIR / "model_comparison_metrics.json"
    summary_df.to_csv(csv_path)
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"\nSaved metrics table to {csv_path}")

    # Plot Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (m_name, cm) in enumerate(aggregated_cms.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        axes[idx].set_title(f"{m_name.replace('_', ' ')}\nTotal Folds Matrix")
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrices.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrices to {cm_path}")

    # 4. SHAP Feature Importance on XGBoost
    print("\nComputing SHAP feature importance for XGBoost on holdout test split...")
    X_train, X_test, y_train, y_test = ml_pipeline.split_data(X, y)
    xgb_final = models["XGBoost"]
    xgb_final.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(xgb_final)
    shap_values = explainer(X_test)
    
    # SHAP Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Feature Importance Summary (XGBoost)", fontsize=14, pad=15)
    plt.tight_layout()
    shap_summary_path = RESULTS_DIR / "shap_summary_xgboost.png"
    plt.savefig(shap_summary_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved SHAP summary plot to {shap_summary_path}")

    # SHAP Mean Absolute Value Bar Plot
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({"Feature": feature_names, "Mean_|SHAP|": mean_abs_shap}).sort_values(by="Mean_|SHAP|", ascending=True)
    
    plt.figure(figsize=(8, 5))
    plt.barh(shap_df["Feature"], shap_df["Mean_|SHAP|"], color="#3b82f6")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    plt.title("XGBoost Feature Importance via SHAP")
    plt.tight_layout()
    shap_bar_path = RESULTS_DIR / "shap_bar_xgboost.png"
    plt.savefig(shap_bar_path, dpi=200)
    plt.close()
    print(f"Saved SHAP bar plot to {shap_bar_path}")

    # Save SHAP rankings to CSV
    shap_csv_path = RESULTS_DIR / "shap_feature_importance.csv"
    shap_df.sort_values(by="Mean_|SHAP|", ascending=False).to_csv(shap_csv_path, index=False)
    print(f"Saved SHAP rankings to {shap_csv_path}")

    print("\n=== P1a Complete Successfully ===")

if __name__ == "__main__":
    run_experiment()
