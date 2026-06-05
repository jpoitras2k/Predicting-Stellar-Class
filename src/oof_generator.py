"""
oof_generator.py — LightGBM Out-of-Fold Probability Generator for Stacking
=============================================================================
Generates clean OOF predictions from a tuned LightGBM model using strict
Stratified K-Fold cross-validation. All fold-level test predictions are
averaged to produce stable test-set probabilities.

Outputs
-------
lgbm_oof_train.csv  — OOF class probabilities for the full training set
                       (ready to drop into a Logistic Regression stacking script)
lgbm_oof_test.csv   — Averaged test-set class probabilities across all folds

Usage
-----
    python src/oof_generator.py
    python src/oof_generator.py --folds 10 --seed 0
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier

# ─── Patch sys.path so we can import from src/ when run from project root ─────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import load_and_preprocess_data, INV_CLASS_MAPPING, CLASS_MAPPING

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
TRAIN_CSV    = ROOT / "Dataset" / "train.csv"
TEST_CSV     = ROOT / "Dataset" / "test.csv"
OOF_TRAIN    = ROOT / "lgbm_oof_train.csv"
OOF_TEST     = ROOT / "lgbm_oof_test.csv"
MODEL_DIR    = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CLASS_NAMES = ["GALAXY", "QSO", "STAR"]
N_CLASSES   = 3

# ─── LightGBM hyperparameters (best configuration from train.py) ─────────────
LGBM_PARAMS = dict(
    n_estimators    = 800,
    learning_rate   = 0.02,
    num_leaves      = 127,
    max_depth       = -1,
    min_child_samples = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq    = 5,
    reg_alpha       = 0.1,
    reg_lambda      = 0.1,
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)


def run(n_folds: int = 5, seed: int = 42) -> None:
    print("\n" + "=" * 60)
    print("  LightGBM OOF Generator")
    print(f"  Folds: {n_folds}  |  Seed: {seed}")
    print("=" * 60 + "\n")

    # ─── Load & preprocess ────────────────────────────────────────────────────
    print("Loading and preprocessing training data...")
    X_train, y_train, ordinal_encoder = load_and_preprocess_data(
        TRAIN_CSV, is_train=True
    )
    print("Loading and preprocessing test data...")
    X_test, _, _ = load_and_preprocess_data(
        TEST_CSV, is_train=False, ordinal_encoder=ordinal_encoder
    )

    # Read IDs for output CSVs
    train_ids = pd.read_csv(TRAIN_CSV, usecols=["id"])["id"].values
    test_ids  = pd.read_csv(TEST_CSV,  usecols=["id"])["id"].values

    print(f"\nFeature dimensions : {X_train.shape[1]} features")
    print(f"Training rows      : {len(y_train):,}")
    print(f"Test rows          : {len(X_test):,}")
    print(f"Class distribution : {dict(zip(CLASS_NAMES, np.bincount(y_train)))}\n")

    # ─── Class weights (corrects STAR imbalance) ──────────────────────────────
    classes      = np.unique(y_train)
    class_weights = dict(zip(
        classes,
        compute_class_weight("balanced", classes=classes, y=y_train)
    ))
    print(f"Class weights: { {INV_CLASS_MAPPING[k]: round(v, 3) for k, v in class_weights.items()} }\n")

    # ─── OOF containers ───────────────────────────────────────────────────────
    oof_train_proba = np.zeros((len(y_train), N_CLASSES), dtype=np.float64)
    oof_test_proba  = np.zeros((len(X_test),  N_CLASSES), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # ─── Cross-validation loop ────────────────────────────────────────────────
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print("-" * 50)
        print(f"  Fold {fold+1} / {n_folds}")
        print("-" * 50)

        X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        # Per-sample class weights for this fold's training split
        sample_weights = np.array([class_weights[c] for c in y_tr])

        # Build and train the model
        model = LGBMClassifier(**LGBM_PARAMS, class_weight=class_weights)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_weights,
        )

        # OOF predictions on the held-out validation fold
        fold_val_proba              = model.predict_proba(X_val)
        oof_train_proba[val_idx]    = fold_val_proba

        # Test predictions for this fold (will be averaged across folds)
        oof_test_proba             += model.predict_proba(X_test)

        fold_loss = log_loss(y_val, fold_val_proba)
        fold_acc  = accuracy_score(y_val, np.argmax(fold_val_proba, axis=1))
        print(f"  [OK] Fold {fold+1}  |  val_loss={fold_loss:.4f}  val_acc={fold_acc:.4f}\n")

    # Average test probabilities across all folds
    oof_test_proba /= n_folds

    # ─── Overall OOF metrics ──────────────────────────────────────────────────
    overall_loss = log_loss(y_train, oof_train_proba)
    overall_acc  = accuracy_score(y_train, np.argmax(oof_train_proba, axis=1))

    print("\n" + "=" * 60)
    print(f"  OVERALL OOF  |  log_loss={overall_loss:.4f}  accuracy={overall_acc:.4f}")
    print("=" * 60)

    print("\nClassification Report (OOF predictions):")
    print(classification_report(
        y_train,
        np.argmax(oof_train_proba, axis=1),
        target_names=CLASS_NAMES
    ))

    # ─── Save OOF train CSV ───────────────────────────────────────────────────
    oof_train_df = pd.DataFrame(
        oof_train_proba,
        columns=[f"lgbm_{c.lower()}_prob" for c in CLASS_NAMES]
    )
    oof_train_df.insert(0, "id", train_ids)
    oof_train_df["lgbm_pred"] = [CLASS_NAMES[i] for i in np.argmax(oof_train_proba, axis=1)]
    oof_train_df.to_csv(OOF_TRAIN, index=False)
    print(f"Saved OOF train probabilities -> {OOF_TRAIN}")

    # ─── Save OOF test CSV ────────────────────────────────────────────────────
    oof_test_df = pd.DataFrame(
        oof_test_proba,
        columns=[f"lgbm_{c.lower()}_prob" for c in CLASS_NAMES]
    )
    oof_test_df.insert(0, "id", test_ids)
    oof_test_df["lgbm_pred"] = [CLASS_NAMES[i] for i in np.argmax(oof_test_proba, axis=1)]
    oof_test_df.to_csv(OOF_TEST, index=False)
    print(f"Saved OOF test probabilities  -> {OOF_TEST}")

    # ─── Prediction distributions ─────────────────────────────────────────────
    print(f"\nTrain prediction distribution:")
    print(f"  {pd.Series(oof_train_df['lgbm_pred']).value_counts().to_dict()}")
    print(f"Test prediction distribution:")
    print(f"  {pd.Series(oof_test_df['lgbm_pred']).value_counts().to_dict()}")

    print("\nDone! Use lgbm_oof_train.csv and lgbm_oof_test.csv in your stacking script.")


# ─── Entry point ──────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LightGBM OOF probability generator")
    p.add_argument("--folds", type=int, default=5,  help="Number of CV folds (default: 5)")
    p.add_argument("--seed",  type=int, default=42, help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(n_folds=args.folds, seed=args.seed)
