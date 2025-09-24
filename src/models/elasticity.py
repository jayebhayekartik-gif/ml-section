# elasticity.py
"""
Price Elasticity Analysis
"""

import pandas as pd
import os

def calculate_price_elasticity(data_path='../../data/processed/pricing_data.csv'):
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    if 'price' not in df.columns or 'demand' not in df.columns:
        print("Required columns not found")
        return

    df['elasticity'] = (df['demand'].pct_change() / df['price'].pct_change()).fillna(0)
    df.to_csv('../../data/processed/elasticity.csv', index=False)
    print("Price elasticity calculated and saved to elasticity.csv")

if __name__ == "__main__":
    calculate_price_elasticity()

