# src/features/build_features.py

import pandas as pd
import logging

def build_features(df, feature_config):
    """
    Build minimal features for ML models:
      - Lag features
      - Rolling averages
      - Optional exogenous features
    """
    logging.info("Building features...")

    df = df.copy()

    col_map = {c.lower(): c for c in df.columns}

    target_col = feature_config.get("target_col", "price")
    target_col = col_map.get(target_col.lower(), list(df.columns)[-1])
    target = df[target_col].copy()

    if "date" in col_map:
        date_col = col_map["date"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    features = pd.DataFrame(index=df.index)

    # Lag features
    for lag in feature_config.get("lag_days", [1]):
        features[f"lag_{lag}"] = target.shift(lag)

    # Rolling mean features
    for window in feature_config.get("rolling_windows", [7]):
        features[f"roll_mean_{window}"] = target.rolling(window, min_periods=1).mean()

    # Exogenous features
    for exog in feature_config.get("exogenous", []):
        if exog.lower() in col_map:
            features[exog] = df[col_map[exog.lower()]]

    # Fill missing values softly (no dropping)
    features = features.bfill().ffill()

    # Keep only rows where target is not NaN
    valid_mask = target.notna()
    features = features.loc[valid_mask]
    target = target.loc[valid_mask]

    logging.info(f"✅ Features built: {features.shape[1]} columns, {features.shape[0]} rows")
    if features.shape[0] == 0:
        logging.warning("⚠️ Warning: No rows left after feature engineering. Check lag/rolling window sizes.")
    return features, target

