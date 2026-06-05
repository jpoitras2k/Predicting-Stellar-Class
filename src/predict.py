import os
import joblib
import numpy as np
import pandas as pd

from preprocess import load_and_preprocess_data, INV_CLASS_MAPPING


def main():
    test_path        = os.path.join('Dataset', 'test.csv')
    encoder_path     = os.path.join('models', 'ordinal_encoder.joblib')
    te_maps_path     = os.path.join('models', 'target_enc_maps.joblib')
    model_names_path = os.path.join('models', 'ensemble_model_names.joblib')

    # Validate saved artefacts exist
    for p in [encoder_path, te_maps_path, model_names_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing artefact: {p}. Please run train.py first.")

    print("Loading test data...")
    test_df  = pd.read_csv(test_path)
    test_ids = test_df['id']

    print("Loading encoder and model list...")
    ordinal_encoder = joblib.load(encoder_path)
    target_enc_maps = joblib.load(te_maps_path)
    model_names     = joblib.load(model_names_path)

    print("Preprocessing test data...")
    X_test, _, _, _ = load_and_preprocess_data(
        test_path, is_train=False, ordinal_encoder=ordinal_encoder,
        target_enc_maps=target_enc_maps
    )

    # ─── Soft-vote ensemble ─────────────────────────────────────────────
    print("Generating ensemble predictions...")
    all_probas = []

    for name in model_names:
        model_path = os.path.join('models', f'{name.lower().replace(" ", "_")}.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        print(f"  Loading {name}...")
        model = joblib.load(model_path)
        proba = model.predict_proba(X_test)
        all_probas.append(proba)

    # Average probabilities across all models
    ensemble_proba = np.mean(all_probas, axis=0)
    preds_int      = np.argmax(ensemble_proba, axis=1)
    mapped_preds   = [INV_CLASS_MAPPING[p] for p in preds_int]

    # ─── Build and validate submission ─────────────────────────────────
    submission = pd.DataFrame({'id': test_ids, 'class': mapped_preds})

    sample_sub_path = os.path.join('Dataset', 'sample_submission.csv')
    if os.path.exists(sample_sub_path):
        sample_sub = pd.read_csv(sample_sub_path)
        print(f"\nSample submission shape : {sample_sub.shape}")
        print(f"Generated submission shape: {submission.shape}")
        if sample_sub.shape == submission.shape:
            print("[OK] Submission shape matches sample submission perfectly!")
        else:
            print("WARNING: Submission shape mismatch!")

    # ─── Ensemble prediction distribution ───────────────────────────────
    pred_dist = pd.Series(mapped_preds).value_counts()
    print(f"\nPrediction distribution:\n{pred_dist.to_string()}")

    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"\nSaved submission to {submission_path}")
    print("\nFirst few rows:")
    print(submission.head(10))


if __name__ == '__main__':
    main()
