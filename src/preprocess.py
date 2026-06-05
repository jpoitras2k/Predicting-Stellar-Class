import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

# Mapping for the target class
CLASS_MAPPING = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
INV_CLASS_MAPPING = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}

def engineer_features(X):
    """
    Adds domain-driven astronomical features to the feature matrix.
    
    Photometric color indices (band differences) capture the spectral energy
    distribution shape, which is the primary physical discriminator between
    GALAXY, QSO, and STAR. Redshift interaction features help separate the
    classes further based on observed vs. intrinsic properties.
    """
    # Photometric color indices (differences between adjacent and broad bands)
    X = X.copy()
    X['u_g']  = X['u'] - X['g']   # UV excess — strong QSO indicator
    X['g_r']  = X['g'] - X['r']   # Core optical slope
    X['r_i']  = X['r'] - X['i']   # Red/blue separation
    X['i_z']  = X['i'] - X['z']   # Near-infrared excess
    X['g_z']  = X['g'] - X['z']   # Broadband color baseline
    X['u_r']  = X['u'] - X['r']   # UV-to-red ratio
    X['u_z']  = X['u'] - X['z']   # Full-span color
    
    # Redshift-based features
    # Stars have near-zero redshift; galaxies moderate; QSOs high
    X['abs_redshift']    = X['redshift'].abs()   # Handle negative noise values
    X['redshift_x_gr']   = X['redshift'] * X['g_r']   # Interaction with color
    X['redshift_x_ur']   = X['redshift'] * X['u_r']   # UV interaction with redshift
    
    # Band magnitude statistics (brightness distribution across bands)
    band_cols = ['u', 'g', 'r', 'i', 'z']
    X['band_mean'] = X[band_cols].mean(axis=1)
    X['band_std']  = X[band_cols].std(axis=1)
    X['band_range'] = X[band_cols].max(axis=1) - X[band_cols].min(axis=1)
    
    return X


def load_and_preprocess_data(file_path, is_train=True, ordinal_encoder=None):
    """
    Loads and preprocesses the dataset.
    
    Parameters:
    - file_path: path to the CSV file.
    - is_train: boolean, whether it is training data.
    - ordinal_encoder: fitted OrdinalEncoder instance (required when is_train=False).
    
    Returns:
    - X: DataFrame of features.
    - y: Series of target labels (if is_train=True, else None).
    - encoder: The fitted OrdinalEncoder instance.
    """
    df = pd.read_csv(file_path)
    
    # Target variable
    y = None
    if is_train and 'class' in df.columns:
        y = df['class'].map(CLASS_MAPPING)
        X = df.drop(columns=['id', 'class'], errors='ignore')
    else:
        X = df.drop(columns=['id'], errors='ignore')
    
    categorical_cols = ['spectral_type', 'galaxy_population']
    
    # Fill missing values for categoricals with a placeholder
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Missing')
    
    # Ordinal encoding for categorical columns
    if is_train:
        ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[categorical_cols] = ordinal_encoder.fit_transform(X[categorical_cols])
    else:
        if ordinal_encoder is None:
            raise ValueError("An ordinal_encoder must be provided for preprocessing test data.")
        X[categorical_cols] = ordinal_encoder.transform(X[categorical_cols])
    
    # Fill any remaining missing values in numeric columns with the column median
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    
    # Apply domain-driven feature engineering
    X = engineer_features(X)
    
    return X, y, ordinal_encoder
