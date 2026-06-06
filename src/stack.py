"""
stack.py — Meta-Learner Stacking Script
=======================================
Loads out-of-fold predictions from all models (oof_train.csv and oof_test.csv)
and trains a Logistic Regression meta-learner to predict the final classes.
Optimizes for Balanced Accuracy.

Outputs
-------
submission_stacked.csv — The final stacked submission file
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

# Map integer targets to string classes
INV_CLASS_MAPPING = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}

def run():
    print("Loading OOF predictions...")
    try:
        train_oof = pd.read_csv('oof_train.csv')
        test_oof  = pd.read_csv('oof_test.csv')
    except FileNotFoundError:
        print("Error: Could not find oof_train.csv or oof_test.csv.")
        print("Make sure to run 'python src/train.py' and 'python src/predict.py' first.")
        return

    # Extract features (probabilities) and target
    X_train = train_oof.drop(columns=['id', 'target'])
    y_train = train_oof['target']
    
    test_ids = test_oof['id']
    X_test   = test_oof.drop(columns=['id'])

    print(f"Train features: {X_train.columns.tolist()}")
    
    # ─── Cross-validate Meta-Learner ────────────────────────────
    print("\nCross-validating Logistic Regression Meta-Learner...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    meta_preds_val = np.zeros(len(train_oof))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr   = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Logistic Regression acts as our Meta-Learner
        meta_model = LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42,
            C=0.1  # Regularization to prevent overfitting to OOFs
        )
        meta_model.fit(X_tr, y_tr)
        
        preds = meta_model.predict(X_val)
        meta_preds_val[val_idx] = preds
        
        b_acc = balanced_accuracy_score(y_val, preds)
        fold_scores.append(b_acc)
        print(f"  Fold {fold+1}: Balanced Accuracy = {b_acc:.4f}")
        
    print(f"\nMeta-Learner CV Balanced Accuracy: {np.mean(fold_scores):.4f}")
    
    print("\nClassification Report (Stacker OOF):")
    print(classification_report(
        y_train, meta_preds_val, 
        target_names=['GALAXY', 'QSO', 'STAR']
    ))
    
    # ─── Final Meta-Learner Training & Prediction ───────────────
    print("\nTraining final Meta-Learner on all OOF data...")
    final_meta = LogisticRegression(
        class_weight='balanced', 
        max_iter=1000, 
        random_state=42,
        C=0.1
    )
    final_meta.fit(X_train, y_train)
    
    # Inspect weights (optional but informative)
    print("\nLearned Model Weights (Coefficients) per Class:")
    weights_df = pd.DataFrame(
        final_meta.coef_, 
        columns=X_train.columns, 
        index=['GALAXY', 'QSO', 'STAR']
    )
    print(weights_df.round(3).to_string())
    
    print("\nGenerating final stacked predictions...")
    stacked_preds = final_meta.predict(X_test)
    mapped_preds = [INV_CLASS_MAPPING[p] for p in stacked_preds]
    
    submission = pd.DataFrame({
        'id': test_ids,
        'class': mapped_preds
    })
    
    submission.to_csv('submission_stacked.csv', index=False)
    print(f"\nSaved final stacked submission to submission_stacked.csv!")
    print(f"Stacked class distribution:\n{pd.Series(mapped_preds).value_counts().to_string()}")

if __name__ == "__main__":
    run()
