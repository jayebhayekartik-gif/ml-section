# build_features.py
# src/features/build_features.py

import pandas as pd
import logging

def build_features(df, feature_config):
    """
    Build minimal features for ML models:
    - Lag features
    - Rolling averages
    """
    logging.info("Building features...")

    features = pd.DataFrame(index=df.index)

    target_col = 'price'  # Replace with your actual target column
    target = df[target_col]

    # Lag features
    for lag in feature_config.get("lag_days", [1]):
        features[f"lag_{lag}"] = target.shift(lag)

    # Rolling mean features
    for window in feature_config.get("rolling_windows", [7]):
        features[f"roll_mean_{window}"] = target.rolling(window).mean()

    # Exogenous features
    for exog in feature_config.get("exogenous", []):
        if exog in df.columns:
            features[exog] = df[exog]

    # Fill NaNs created by shift/rolling
    features.fillna(method='bfill', inplace=True)

    logging.info(f"Features built: {features.shape[1]} columns")
    return features, target

