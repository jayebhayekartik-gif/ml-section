# src/features/build_features.py

import pandas as pd
import logging

def build_features(df, feature_config):
    """
    Build minimal features for ML models:
      - Lag features
      - Rolling averages
      - Optional exogenous features

    Returns:
      features (DataFrame), target (Series)
    """
    logging.info("Building features...")

    # Work on a copy to avoid chained-assignment warnings
    df = df.copy()

    # --- column name handling (case-insensitive) ---
    col_map = {c.lower(): c for c in df.columns}

    # Determine target column (config can override)
    requested_target = feature_config.get("target_col", "price")
    if requested_target.lower() in col_map:
        target_col = col_map[requested_target.lower()]
    elif "price" in col_map:
        target_col = col_map["price"]
    else:
        raise KeyError("Target column 'price' (or config-specified) not found in dataframe columns")

    # If there's a date column, make sure data is sorted by it
    if "date" in col_map:
        date_col = col_map["date"]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # Prepare target series
    target = df[target_col].copy()

    # Create features DataFrame aligned with df index
    features = pd.DataFrame(index=df.index)

    # --- Lag features ---
    lag_days = feature_config.get("lag_days", [1])
    # accept single int or list
    if isinstance(lag_days, int):
        lag_days = [lag_days]

    for lag in lag_days:
        features[f"lag_{lag}"] = target.shift(lag)

    # --- Rolling mean features ---
    rolling_windows = feature_config.get("rolling_windows", feature_config.get("rolling_window", [7]))
    if isinstance(rolling_windows, int):
        rolling_windows = [rolling_windows]

    for window in rolling_windows:
        # require window > 0
        if window > 0:
            features[f"roll_mean_{window}"] = target.rolling(window=window, min_periods=1).mean()

    # --- Exogenous features (case-insensitive matching) ---
    for exog in feature_config.get("exogenous", []):
        if exog.lower() in col_map:
            features[exog] = df[col_map[exog.lower()]]
        else:
            logging.debug(f"Exogenous feature '{exog}' not found in dataframe; skipping.")

    # --- Clean up NaNs created by shifts/rolling ---
    # Prefer dropping rows with NaNs (safer for training) after trying to fill
    # First do a sensible fill for edges, then drop remaining NaNs
    features = features.fillna(method="bfill").fillna(method="ffill")
    features = features.dropna(how="any")

    # Align target to the final features' index
    target = target.loc[features.index]

    logging.info(f"Features built: {features.shape[1]} columns, {features.shape[0]} rows")
    return features, target


