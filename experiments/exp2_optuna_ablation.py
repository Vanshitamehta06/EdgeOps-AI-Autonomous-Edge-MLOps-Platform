"""
experiments/exp2_optuna_ablation.py

P1b: Optuna Hyperparameter Optimization Ablation Study.
Quantitatively compares:
1. Default Vanilla XGBoost (unweighted, standard parameters)
2. Baseline Balanced XGBoost (dynamic scale_pos_weight, fixed parameters)
3. Optuna Tuned XGBoost (30 Bayesian optimization trials over parameter space)

Evaluates on a stratified holdout test split (80/20) for fair comparison.
Reports: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, and Optimization trajectory.
Outputs saved to results/.
"""

import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import config
import ml_pipeline

plt.switch_backend('Agg')
optuna.logging.set_verbosity(optuna.logging.WARNING)

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_TRIALS = 30

def run_experiment():
    print(f"=== P1b: Optuna Ablation Study on {config.DATASET_PATH.name} ===")
    
    # 1. Load and split data
    df = pd.read_csv(config.DATASET_PATH)
    X, y, encoder = ml_pipeline.preprocess_data(df)
    X_train, X_test, y_train, y_test = ml_pipeline.split_data(X, y)
    
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    dynamic_weight = float(neg_count / max(pos_count, 1))
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)} | Dynamic weight: {dynamic_weight:.2f}")

    results = {}

    # -------------------------------------------------------------
    # CONFIG 1: Vanilla Default XGBoost
    # -------------------------------------------------------------
    print("\n[1/3] Training Vanilla Default XGBoost (unweighted)...")
    t0 = time.perf_counter()
    vanilla_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    vanilla_model.fit(X_train, y_train)
    vanilla_time = time.perf_counter() - t0
    
    y_pred_v = vanilla_model.predict(X_test)
    y_prob_v = vanilla_model.predict_proba(X_test)[:, 1]
    
    results["Vanilla_Default"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_v)),
        "precision": float(precision_score(y_test, y_pred_v, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_v, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_v, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_v)),
        "pr_auc": float(average_precision_score(y_test, y_prob_v)),
        "train_time_sec": float(vanilla_time),
        "params": vanilla_model.get_params()
    }
    print(f" -> Vanilla F1: {results['Vanilla_Default']['f1']:.4f} | Recall: {results['Vanilla_Default']['recall']:.4f} | ROC-AUC: {results['Vanilla_Default']['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # CONFIG 2: Baseline Balanced XGBoost
    # -------------------------------------------------------------
    print("\n[2/3] Training Baseline Balanced XGBoost (dynamic scale_pos_weight)...")
    t0 = time.perf_counter()
    baseline_params = config.XGB_PARAMS.copy()
    baseline_params["scale_pos_weight"] = dynamic_weight
    balanced_model = XGBClassifier(**baseline_params)
    balanced_model.fit(X_train, y_train)
    balanced_time = time.perf_counter() - t0
    
    y_pred_b = balanced_model.predict(X_test)
    y_prob_b = balanced_model.predict_proba(X_test)[:, 1]
    
    results["Baseline_Balanced"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_b)),
        "precision": float(precision_score(y_test, y_pred_b, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_b, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_b, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_b)),
        "pr_auc": float(average_precision_score(y_test, y_prob_b)),
        "train_time_sec": float(balanced_time),
        "params": baseline_params
    }
    print(f" -> Baseline Balanced F1: {results['Baseline_Balanced']['f1']:.4f} | Recall: {results['Baseline_Balanced']['recall']:.4f} | ROC-AUC: {results['Baseline_Balanced']['roc_auc']:.4f}")

    # -------------------------------------------------------------
    # CONFIG 3: Optuna Bayesian Hyperparameter Optimization
    # -------------------------------------------------------------
    print(f"\n[3/3] Running Optuna Bayesian HPO ({N_TRIALS} trials)...")
    trial_scores = []
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.30, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", dynamic_weight * 0.5, dynamic_weight * 1.5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, zero_division=0)
        trial_scores.append(f1)
        return f1

    t0_opt = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_TRIALS)
    total_optuna_time = time.perf_counter() - t0_opt

    print(f"Optuna completed in {total_optuna_time:.2f}s. Best validation F1: {study.best_value:.4f}")
    print(f"Best hyperparameters: {study.best_params}")

    best_optuna_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        **study.best_params
    }
    
    t0_fit = time.perf_counter()
    tuned_model = XGBClassifier(**best_optuna_params)
    tuned_model.fit(X_train, y_train)
    tuned_fit_time = time.perf_counter() - t0_fit
    
    y_pred_t = tuned_model.predict(X_test)
    y_prob_t = tuned_model.predict_proba(X_test)[:, 1]

    results["Optuna_Tuned"] = {
        "accuracy": float(accuracy_score(y_test, y_pred_t)),
        "precision": float(precision_score(y_test, y_pred_t, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_t, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_t, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob_t)),
        "pr_auc": float(average_precision_score(y_test, y_prob_t)),
        "train_time_sec": float(tuned_fit_time),
        "total_hpo_time_sec": float(total_optuna_time),
        "params": best_optuna_params
    }
    print(f" -> Optuna Tuned F1: {results['Optuna_Tuned']['f1']:.4f} | Recall: {results['Optuna_Tuned']['recall']:.4f} | ROC-AUC: {results['Optuna_Tuned']['roc_auc']:.4f}")

    # Save summary table
    summary_rows = []
    for model_name, data in results.items():
        summary_rows.append({
            "Configuration": model_name,
            "Accuracy": f"{data['accuracy']:.4f}",
            "Precision": f"{data['precision']:.4f}",
            "Recall": f"{data['recall']:.4f}",
            "F1_Score": f"{data['f1']:.4f}",
            "ROC_AUC": f"{data['roc_auc']:.4f}",
            "PR_AUC": f"{data['pr_auc']:.4f}",
            "Fit_Time_Sec": f"{data['train_time_sec']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = RESULTS_DIR / "optuna_ablation_metrics.csv"
    summary_df.to_csv(csv_path, index=False)
    
    with open(RESULTS_DIR / "optuna_ablation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nSaved ablation metrics to {csv_path}")

    # Plot Optimization Trajectory
    plt.figure(figsize=(10, 5))
    trial_values = [t.value for t in study.trials if t.value is not None]
    best_running = np.maximum.accumulate(trial_values)
    plt.plot(range(1, len(trial_values) + 1), trial_values, 'o-', color='#93c5fd', alpha=0.6, label='Trial Score (F1)')
    plt.plot(range(1, len(best_running) + 1), best_running, 'r-', linewidth=2, label='Best Running F1')
    plt.axhline(results["Vanilla_Default"]["f1"], color='gray', linestyle='--', label=f'Vanilla Default ({results["Vanilla_Default"]["f1"]:.4f})')
    plt.axhline(results["Baseline_Balanced"]["f1"], color='green', linestyle=':', label=f'Baseline Balanced ({results["Baseline_Balanced"]["f1"]:.4f})')
    plt.xlabel("Optuna Trial Iteration")
    plt.ylabel("Holdout Test F1-Score")
    plt.title("Optuna Bayesian Optimization Trajectory (F1-Score)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    trajectory_path = RESULTS_DIR / "optuna_optimization_history.png"
    plt.savefig(trajectory_path, dpi=200)
    plt.close()
    print(f"Saved optimization history plot to {trajectory_path}")

    print("\n=== P1b Complete Successfully ===")

if __name__ == "__main__":
    run_experiment()
