
# train.py
"""
Train ML models for Smart Pricing Intelligence Platform
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression(data_path='../../data/processed/pricing_data.csv', model_path='linear_model.pkl'):
    # Load dataset
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Example: using price as target, features can be extended
    if 'price' not in df.columns or 'feature1' not in df.columns:
        print("Required columns not found in data")
        return

    X = df[['feature1']].values
    y = df['price'].values

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_linear_regression()


