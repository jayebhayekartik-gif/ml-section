
# evaluate.py
"""
Evaluate ML models
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

def evaluate_model(data_path='../../data/processed/pricing_data.csv', model_path='linear_model.pkl'):
    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Data file or model not found")
        return

    df = pd.read_csv(data_path)
    X = df[['feature1']].values
    y = df['price'].values

    model = joblib.load(model_path)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

if __name__ == "__main__":
    evaluate_model()

