"""
tune_xgb.py — Optuna Hyperparameter Tuning Script for XGBoost
=============================================================
Automated hyperparameter search using Optuna for XGBoost. 
Optimizes for Balanced Accuracy using 3-fold stratified CV.

Usage
-----
    python src/tune_xgb.py --trials 50
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

# Patch sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_and_preprocess_data

ROOT      = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "Dataset" / "train.csv"
TUNING_DIR = ROOT / "tuning"
TUNING_DIR.mkdir(exist_ok=True)

SEED = 42
N_FOLDS = 3

def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
        'tree_method': 'hist' # Faster histogram based training
    }
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    
    global X_train, y_train_num, class_weights_dict
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train_num)):
        X_tr, y_tr   = X_train.iloc[tr_idx], y_train_num.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train_num.iloc[val_idx]

        # XGBoost handles sample weights via fit()
        sample_weights = np.array([class_weights_dict[c] for c in y_tr])
        
        model = XGBClassifier(**param)
        
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights
        )
        
        preds = model.predict_proba(X_val)
        pred_labels = np.argmax(preds, axis=1)
        bal_acc = balanced_accuracy_score(y_val, pred_labels)
        fold_scores.append(bal_acc)
        
    return 1 - np.mean(fold_scores)  # minimize 1 - balanced_accuracy


def run(n_trials: int):
    print("\n" + "="*50)
    print("  Optuna Hyperparameter Tuning - XGBoost")
    print("  Metric: Balanced Accuracy (maximizing)")
    print(f"  Target Trials: {n_trials}")
    print("="*50 + "\n")
    
    global X_train, y_train_num, class_weights_dict
    print("Loading data...")
    X_train, y_train_raw, _, _ = load_and_preprocess_data(TRAIN_CSV, is_train=True)
    
    # XGBoost requires numeric targets 0, 1, 2
    classes = np.unique(y_train_raw)
    target_map = {c: i for i, c in enumerate(classes)}
    y_train_num = y_train_raw.map(target_map)
    
    classes_num = np.unique(y_train_num)
    cw_array = compute_class_weight("balanced", classes=classes_num, y=y_train_num)
    class_weights_dict = dict(zip(classes_num, cw_array))
    
    study_name = "xgb_balanced_acc"
    db_path = TUNING_DIR / f"{study_name}.db"
    
    study = optuna.create_study(
        direction="minimize", 
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True
    )
    
    print("\nStarting optimization...")
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user. Best results saved.")
        
    print("\n" + "="*50)
    print("  TUNING COMPLETE")
    print("="*50)
    best_bal_acc = 1 - study.best_value
    print(f"  Best Balanced Accuracy : {best_bal_acc:.4f}")
    print("  Best Params   :")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    best_params_path = TUNING_DIR / "xgb_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"\nSaved best parameters to {best_params_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    args = parser.parse_args()
    
    run(n_trials=args.trials)
