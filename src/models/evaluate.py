# src/models/evaluate.py
import logging
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_predictions(y_true, y_pred):
    """Compute MAE, MSE, RMSE, and R² metrics safely."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}")

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"MAE": float(mae), "MSE": float(mse), "RMSE": float(rmse), "R2": float(r2)}


def evaluate_model(data_path, model_path, feature_cols=None, target_col="Price"):
    """
    Load data and trained model, make predictions, and log evaluation metrics.
    Automatically handles missing or extra feature columns.
    """
    # --- 1. Validate paths ---
    if not os.path.exists(data_path):
        logging.error(f"Data file not found: {data_path}")
        return
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    # --- 2. Load CSV ---
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        logging.error(f"Target column '{target_col}' not found in data. Available columns: {df.columns.tolist()}")
        return

    # --- 3. Load model ---
    model = joblib.load(model_path)

    # Get feature names model was trained with (if available)
    trained_features = getattr(model, "feature_names_in_", None)

    # --- 4. Determine current features ---
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.lower() not in {target_col.lower(), "date"}]

    # --- 5. Align current dataset with trained features ---
    if trained_features is not None:
        # Add missing features with 0 or mean value
        for feat in trained_features:
            if feat not in df.columns:
                logging.warning(f"Adding missing feature '{feat}' as zeros for evaluation.")
                df[feat] = 0

        # Drop any extra columns not seen during training
        extra = [c for c in df.columns if c not in trained_features and c != target_col]
        if extra:
            logging.info(f"Dropping extra columns not seen at training time: {extra}")
        feature_cols = trained_features

    # --- 6. Prepare X and y ---
    X = df[feature_cols]
    y = df[target_col].values

    # --- 7. Predict and evaluate ---
    preds = model.predict(X)
    metrics = evaluate_predictions(y, preds)

    logging.info(f"✅ Evaluation completed successfully.")
    logging.info(f"📊 Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    evaluate_model(
        data_path="D:/shantanu/git/ml-section/data/processed/processed_pp_prices_clean.csv",
        model_path="D:/shantanu/git/ml-section/models/linear_model.pkl",
        feature_cols=None,
        target_col="Price"
    )





