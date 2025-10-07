# src/pipeline.py
"""
Smart Pricing Intelligence Platform - End-to-End Pipeline
This script ties together:
1. Data loading
2. Preprocessing
3. Feature engineering
4. Model training (SARIMAX + XGBoost)
5. Evaluation (MAE, RMSE, MAPE)
6. Saving models and predictions
"""

import os
import yaml
import logging
import pandas as pd
from src.data.loader import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_xgboost, train_sarimax
from src.models.evaluate import evaluate_predictions
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -----------------------------
# Load configuration
# -----------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# -----------------------------
# Step 1: Load Data
# -----------------------------
logging.info("Loading raw data...")
data = load_data(config["data"]["raw_path"])

# -----------------------------
# Step 2: Preprocess Data
# -----------------------------
logging.info("Preprocessing data...")
clean_data = preprocess_data(data)

# -----------------------------
# Step 3: Build Features
# -----------------------------
logging.info("Building features...")
features, target = build_features(clean_data, config["features"])

# -----------------------------
# Step 4: Train Models
# -----------------------------
# XGBoost
logging.info("Training XGBoost model...")
xgb_model = train_xgboost(features, target, config["models"]["xgboost"])
joblib.dump(xgb_model, os.path.join(config["models"]["save_path"], "xgboost_model.pkl"))

# SARIMAX
logging.info("Training SARIMAX model...")
sarimax_model, sarimax_pred = train_sarimax(features, target, config["models"]["sarimax"])
joblib.dump(sarimax_model, os.path.join(config["models"]["save_path"], "sarimax_model.pkl"))

# -----------------------------
# Step 5: Evaluate Models
# -----------------------------
logging.info("Evaluating XGBoost model...")
xgb_pred = xgb_model.predict(features)
xgb_metrics = evaluate_predictions(target, xgb_pred)

logging.info("Evaluating SARIMAX model...")
sarimax_metrics = evaluate_predictions(target, sarimax_pred)

logging.info(f"XGBoost Metrics: {xgb_metrics}")
logging.info(f"SARIMAX Metrics: {sarimax_metrics}")

# -----------------------------
# Step 6: Save Predictions
# -----------------------------
predictions_df = pd.DataFrame({
    "actual": target,
    "xgb_pred": xgb_pred,
    "sarimax_pred": sarimax_pred
})
predictions_df.to_csv(os.path.join(config["data"]["processed_path"], "predictions.csv"), index=False)

# -----------------------------
# Step 7: Simple Visualization
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(predictions_df["actual"], label="Actual", color="black")
plt.plot(predictions_df["xgb_pred"], label="XGBoost", linestyle="--")
plt.plot(predictions_df["sarimax_pred"], label="SARIMAX", linestyle=":")
plt.xlabel("Time")
plt.ylabel("Price")
plt.title("Predicted vs Actual Prices")
plt.legend()
plt.savefig(os.path.join(config["reports"]["plots_path"], "predicted_vs_actual.png"))
plt.show()

logging.info("Pipeline completed successfully!")
