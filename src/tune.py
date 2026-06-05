"""
tune.py — Optuna Hyperparameter Tuning Script
=============================================
Automated hyperparameter search using Optuna. Currently configured to optimize
LightGBM using a 3-fold stratified cross-validation. Optimization target is
Log Loss (minimization).

Usage
-----
    python src/tune.py --trials 50
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
from sklearn.metrics import log_loss, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

# Patch sys.path so we can import from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_and_preprocess_data

ROOT      = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "Dataset" / "train.csv"
TUNING_DIR = ROOT / "tuning"
TUNING_DIR.mkdir(exist_ok=True)

# Fixed configurations
SEED = 42
N_FOLDS = 3  # 3 is a good balance between speed and reliability for tuning

def objective(trial):
    """Optuna objective function for LightGBM."""
    
    # 1. Define hyperparameter search space
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 256),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    # 2. Setup Cross-Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    
    # We load data inside run() as a global to avoid loading on every trial
    global X_train, y_train, class_weights
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        sample_weights = np.array([class_weights[c] for c in y_tr])
        
        model = LGBMClassifier(**param, class_weight=class_weights)
        
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
    print("  Optuna Hyperparameter Tuning - LightGBM")
    print("  Metric: Balanced Accuracy (maximizing)")
    print(f"  Target Trials: {n_trials}")
    print("="*50 + "\n")
    
    # Load dataset globally for the objective function
    global X_train, y_train, class_weights
    print("Loading data...")
    X_train, y_train, _, _ = load_and_preprocess_data(TRAIN_CSV, is_train=True)
    
    classes = np.unique(y_train)
    class_weights = dict(zip(
        classes,
        compute_class_weight("balanced", classes=classes, y=y_train)
    ))
    
    # Create study
    study_name = "lgbm_balanced_acc"
    db_path = TUNING_DIR / f"{study_name}.db"
    
    # We use sqlite so we can stop/resume safely
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
        
    # Save best parameters to JSON
    best_params_path = TUNING_DIR / "lgbm_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"\nSaved best parameters to {best_params_path}")
    print("You can copy these parameters into src/train.py or src/oof_generator.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    args = parser.parse_args()
    
    run(n_trials=args.trials)
