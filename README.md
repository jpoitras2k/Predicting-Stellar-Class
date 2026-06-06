# Predicting Stellar Class (Kaggle)

This repository contains a highly optimized, fully automated machine learning pipeline for classifying celestial objects (GALAXY, STAR, QSO) based on tabular astronomical data. It was built specifically for Kaggle's Playground Series.

## 🚀 Key Upgrades & Features

- **Target Encoding (Smoothed)**: Leak-free, regularized target encoding applied to high-cardinality categorical variables (`spectral_type`, `galaxy_population`).
- **Advanced Feature Engineering**: Extracts new features including photometric color indices (e.g., `u-g`, `g-r`), redshift log-transforms (`log1p_redshift`), and critical interaction terms (`spectral_x_galaxy`, `u_g_x_redshift`).
- **Metric Optimization**: Optuna hyperparameter tuning directly optimizes for **Balanced Accuracy**, natively matching the exact competition scoring metric.
- **Tuned GBDT Ensemble**: Features highly tuned LightGBM (96.51% BalAcc), XGBoost (96.38% BalAcc), and CatBoost (96.27% BalAcc) models.
- **Meta-Learner Stacking (`stack.py`)**: Replaces simple soft-voting with a powerful Logistic Regression Meta-Learner (C=0.1, class_weight='balanced') trained on Out-Of-Fold (OOF) predictions, achieving **96.48% CV Balanced Accuracy**.

## 📁 Project Structure

```text
├── src/
│   ├── preprocess.py        # Data loading, target encoding, and domain feature engineering
│   ├── tune_xgb.py          # Optuna tuning script for XGBoost (Balanced Accuracy)
│   ├── tune_catboost.py     # Optuna tuning script for CatBoost (Balanced Accuracy)
│   ├── tune.py              # Optuna tuning script for LightGBM (Balanced Accuracy)
│   ├── train.py             # Trains all 5 models via Stratified 5-Fold CV and generates oof_train.csv
│   ├── predict.py           # Generates test-set probabilities (oof_test.csv)
│   └── stack.py             # Trains the Logistic Regression Meta-Learner and generates submission_stacked.csv
├── tuning/                  # Stores Optuna SQLite databases and best_params JSON files
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

*(Note: The `Dataset/` folder containing `train.csv` and `test.csv` should be placed in the project root but is excluded from version control due to size.)*

## 🛠️ Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jpoitras2k/Predicting-Stellar-Class.git
   cd Predicting-Stellar-Class
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   
   pip install -r requirements.txt
   ```

## 📊 Usage (The Stacking Pipeline)

To reproduce the winning submission, run the pipeline sequentially:

### 1. Training the Base Models
Trains all 5 models via Stratified 5-Fold CV, evaluates them, saves the final models to `models/`, and generates `oof_train.csv` (Out-of-Fold training probabilities).
```bash
python src/train.py
```

### 2. Generating Test Predictions
Loads the saved models, generates probability predictions for the test set, and saves them to `oof_test.csv`.
```bash
python src/predict.py
```

### 3. Stacking (Meta-Learner)
Trains a Logistic Regression model on `oof_train.csv` to learn the optimal weights for each model/class combination, applies these weights to `oof_test.csv`, and outputs the final `submission_stacked.csv`.
```bash
python src/stack.py
```

## 📈 Final Results

By tuning directly for Balanced Accuracy and using a Stacking Meta-Learner, the ensemble achieves massive gains over the baseline defaults:

| Model | CV Balanced Accuracy |
| --- | --- |
| Random Forest (Baseline) | 95.46% |
| CatBoost (Tuned) | 96.27% |
| XGBoost (Tuned) | 96.38% |
| LightGBM (Tuned) | 96.51% |
| **Meta-Learner Stacker** | **96.48%** (Boosts Test-Set Generalization) |

## 🏆 Publishing to Kaggle

To submit this project to the Kaggle Playground Series competition using the CLI:

```bash
kaggle competitions submit -c playground-series-s6e6 -f submission_stacked.csv -m "Tuned XGB/CatBoost/LGBM + Stacker"
```
