import pandas as pd
import os

def calculate_price_elasticity(
    data_path='D:/shantanu/git/ml-section/data/processed/processed_pp_prices_clean.csv',
    output_path='D:/shantanu/git/ml-section/data/processed/elasticity.csv'
):
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)

    # Create synthetic demand if not present (for testing)
    if 'demand' not in df.columns:
        # Example: demand inversely proportional to price
        df['demand'] = 1000 / df['Price']

    # Compute price elasticity
    df['elasticity'] = (df['demand'].pct_change() / df['Price'].pct_change()).fillna(0)
    df.to_csv(output_path, index=False)
    print(f"✅ Price elasticity calculated and saved to {output_path}")

if __name__ == "__main__":
    calculate_price_elasticity()


