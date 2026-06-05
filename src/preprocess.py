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
    X['log1p_redshift']  = np.log1p(X['redshift'].clip(lower=0))  # Log transform
    X['u_g_x_redshift']  = X['u_g'] * X['redshift']   # UV excess x redshift
    
    # Band magnitude statistics (brightness distribution across bands)
    band_cols = ['u', 'g', 'r', 'i', 'z']
    X['band_mean'] = X[band_cols].mean(axis=1)
    X['band_std']  = X[band_cols].std(axis=1)
    X['band_range'] = X[band_cols].max(axis=1) - X[band_cols].min(axis=1)
    
    return X


def apply_target_encoding(X, y=None, target_enc_maps=None, cat_cols=None):
    """
    Apply smoothed target encoding for categorical columns.
    
    During training (y is not None): computes the mean of numeric target per
    category with global prior smoothing to prevent overfitting on rare categories.
    
    During inference (target_enc_maps provided): applies the stored mappings.
    
    Returns the modified X and the encoding maps.
    """
    if cat_cols is None:
        cat_cols = ['spectral_type', 'galaxy_population']
    
    smoothing = 10  # regularization strength
    
    if y is not None and target_enc_maps is None:
        # Training: compute target encoding maps
        target_enc_maps = {}
        global_mean = y.mean()
        
        for col in cat_cols:
            if col not in X.columns:
                continue
            stats = y.groupby(X[col]).agg(['mean', 'count'])
            # Smoothed encoding: blend category mean with global mean
            smoother = 1 / (1 + np.exp(-(stats['count'] - smoothing) / smoothing))
            stats['smoothed'] = smoother * stats['mean'] + (1 - smoother) * global_mean
            target_enc_maps[col] = stats['smoothed'].to_dict()
            X[f'{col}_te'] = X[col].map(target_enc_maps[col]).fillna(global_mean)
        
        # Interaction: spectral_type x galaxy_population
        if 'spectral_type' in X.columns and 'galaxy_population' in X.columns:
            interaction_col = X['spectral_type'].astype(str) + '_' + X['galaxy_population'].astype(str)
            stats = y.groupby(interaction_col).agg(['mean', 'count'])
            smoother = 1 / (1 + np.exp(-(stats['count'] - smoothing) / smoothing))
            stats['smoothed'] = smoother * stats['mean'] + (1 - smoother) * global_mean
            target_enc_maps['spectral_x_galaxy'] = stats['smoothed'].to_dict()
            X['spectral_x_galaxy_te'] = interaction_col.map(target_enc_maps['spectral_x_galaxy']).fillna(global_mean)
    
    elif target_enc_maps is not None:
        # Inference: apply stored maps
        for col in cat_cols:
            if col not in X.columns or col not in target_enc_maps:
                continue
            global_mean = np.mean(list(target_enc_maps[col].values()))
            X[f'{col}_te'] = X[col].map(target_enc_maps[col]).fillna(global_mean)
        
        if 'spectral_x_galaxy' in target_enc_maps:
            if 'spectral_type' in X.columns and 'galaxy_population' in X.columns:
                interaction_col = X['spectral_type'].astype(str) + '_' + X['galaxy_population'].astype(str)
                global_mean = np.mean(list(target_enc_maps['spectral_x_galaxy'].values()))
                X['spectral_x_galaxy_te'] = interaction_col.map(target_enc_maps['spectral_x_galaxy']).fillna(global_mean)
    
    return X, target_enc_maps


def load_and_preprocess_data(file_path, is_train=True, ordinal_encoder=None,
                             target_enc_maps=None):
    """
    Loads and preprocesses the dataset.
    
    Parameters:
    - file_path: path to the CSV file.
    - is_train: boolean, whether it is training data.
    - ordinal_encoder: fitted OrdinalEncoder instance (required when is_train=False).
    - target_enc_maps: dict of target encoding maps (required when is_train=False).
    
    Returns:
    - X: DataFrame of features.
    - y: Series of target labels (if is_train=True, else None).
    - encoder: The fitted OrdinalEncoder instance.
    - target_enc_maps: dict of target encoding maps.
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
    
    # Target encoding (before ordinal encoding so we can use string categories)
    if is_train:
        X, target_enc_maps = apply_target_encoding(X, y=y, cat_cols=categorical_cols)
    else:
        X, _ = apply_target_encoding(X, target_enc_maps=target_enc_maps, cat_cols=categorical_cols)
    
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
    
    return X, y, ordinal_encoder, target_enc_maps

