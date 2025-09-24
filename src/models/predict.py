
# predict.py
"""
Generate pricing predictions
"""

import pandas as pd
import joblib
import os

def generate_predictions(input_csv='../../data/processed/new_products.csv', model_path='linear_model.pkl', output_csv='predictions.csv'):
    if not os.path.exists(input_csv) or not os.path.exists(model_path):
        print("Input file or model not found")
        return

    df = pd.read_csv(input_csv)
    model = joblib.load(model_path)

    df['predicted_price'] = model.predict(df[['feature1']])
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    generate_predictions()
