import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from preprocess import load_and_preprocess_data, INV_CLASS_MAPPING

N_FOLDS = 5
RANDOM_STATE = 42
N_CLASSES = 3


def compute_class_weights(y):
    """Compute balanced class weights to handle STAR class imbalance."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))


def get_models(class_weights):
    """Return all model definitions with tuned hyperparameters."""
    sample_weight_map = class_weights  # used for non-sklearn native models

    return {
        'LightGBM': LGBMClassifier(
            n_estimators=800,
            learning_rate=0.02,
            num_leaves=127,
            max_depth=-1,
            min_child_samples=20,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight=class_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        'XGBoost': XGBClassifier(
            n_estimators=600,
            learning_rate=0.02,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric='mlogloss',
            verbosity=0,
        ),
        'CatBoost': CatBoostClassifier(
            iterations=600,
            learning_rate=0.05,
            depth=8,
            l2_leaf_reg=3,
            auto_class_weights='Balanced',
            random_seed=RANDOM_STATE,
            verbose=0,
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def main():
    os.makedirs('models', exist_ok=True)

    train_path = os.path.join('Dataset', 'train.csv')
    print(f"Loading data from {train_path}...")
    X, y, ordinal_encoder = load_and_preprocess_data(train_path, is_train=True)

    print(f"Feature set size: {X.shape[1]} features, {X.shape[0]} rows")
    print(f"Class distribution:\n{y.value_counts().rename({0:'GALAXY',1:'QSO',2:'STAR'})}\n")

    # Save encoder for use at prediction time
    encoder_path = os.path.join('models', 'ordinal_encoder.joblib')
    joblib.dump(ordinal_encoder, encoder_path)
    print(f"Saved OrdinalEncoder to {encoder_path}")

    # Compute class weights for STAR imbalance
    class_weights = compute_class_weights(y.values)
    print(f"Class weights: { {INV_CLASS_MAPPING[k]: round(v, 3) for k, v in class_weights.items()} }\n")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models = get_models(class_weights)
    model_names = list(models.keys())

    # Store per-model OOF probability arrays
    oof_probas = {name: np.zeros((len(X), N_CLASSES)) for name in model_names}
    results = {}

    # ─── Per-model cross-validation ───────────────────────────────────────────
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Evaluating: {name}")
        print(f"{'='*50}")
        fold_accs, fold_losses = [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val,   y_val   = X.iloc[val_idx],   y.iloc[val_idx]

            # XGBoost handles class imbalance via sample_weight
            if name == 'XGBoost':
                sw = np.array([class_weights[c] for c in y_train])
                model.fit(X_train, y_train, sample_weight=sw)
            else:
                model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)
            val_preds = np.argmax(val_proba, axis=1)

            oof_probas[name][val_idx] = val_proba

            acc  = accuracy_score(y_val, val_preds)
            loss = log_loss(y_val, val_proba)
            fold_accs.append(acc)
            fold_losses.append(loss)
            print(f"  Fold {fold+1}/{N_FOLDS}: Accuracy={acc:.4f}  LogLoss={loss:.4f}")

        mean_acc  = np.mean(fold_accs)
        mean_loss = np.mean(fold_losses)
        results[name] = {'Accuracy': mean_acc, 'Log Loss': mean_loss}
        print(f"  >> Mean Accuracy: {mean_acc:.4f}  |  Mean Log Loss: {mean_loss:.4f}")

    # ─── Ensemble OOF evaluation ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  Evaluating: Soft Ensemble (equal weights)")
    print(f"{'='*50}")
    ensemble_proba = np.mean([oof_probas[n] for n in model_names], axis=0)
    ensemble_preds = np.argmax(ensemble_proba, axis=1)
    ens_acc  = accuracy_score(y, ensemble_preds)
    ens_loss = log_loss(y, ensemble_proba)
    results['Soft Ensemble'] = {'Accuracy': ens_acc, 'Log Loss': ens_loss}
    print(f"  >> OOF Accuracy: {ens_acc:.4f}  |  OOF Log Loss: {ens_loss:.4f}")

    # ─── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(results).T.sort_values('Accuracy', ascending=False)
    print(summary_df.to_string())
    print(f"{'='*60}")

    # ─── Retrain all models on full dataset and save ───────────────────────────
    print("\nRetraining all models on the full dataset for final submission...")
    final_models = get_models(class_weights)

    for name, model in final_models.items():
        print(f"  Training {name}...")
        if name == 'XGBoost':
            sw = np.array([class_weights[c] for c in y])
            model.fit(X, y, sample_weight=sw)
        else:
            model.fit(X, y)
        save_path = os.path.join('models', f'{name.lower().replace(" ", "_")}.joblib')
        joblib.dump(model, save_path)
        print(f"  Saved to {save_path}")

    # Save the list of model names used for ensemble
    joblib.dump(model_names, os.path.join('models', 'ensemble_model_names.joblib'))

    # Classification report of the ensemble OOF predictions
    print("\nClassification Report (Ensemble OOF Predictions):")
    print(classification_report(
        y, ensemble_preds,
        target_names=[INV_CLASS_MAPPING[i] for i in range(N_CLASSES)]
    ))
    print("Done! All models saved to models/")


if __name__ == '__main__':
    main()
