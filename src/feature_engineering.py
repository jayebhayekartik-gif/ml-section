import pandas as pd

# Load cleaned CSVs
pp = pd.read_csv("../data/processed/processed_pp_prices_clean.csv")
naphtha = pd.read_csv("../data/processed/processed_naphtha_prices_clean.csv")
propylene = pd.read_csv("../data/processed/processed_propylene_prices_clean.csv")
freight = pd.read_csv("../data/processed/processed_freight_cost_clean.csv")

# Convert Date to datetime
for df in [pp, naphtha, propylene, freight]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Merge dataframes on Date
df = pp.merge(naphtha, on='Date', suffixes=('_pp', '_naphtha'))
df = df.merge(propylene, on='Date')
df = df.merge(freight, on='Date', suffixes=('', '_freight'))

# Create spreads
df['naphtha_propylene_spread'] = df['Price'] - df['Price_naphtha']
df['propylene_pp_spread'] = df['Price_pp'] - df['Price']

# Create lag features for PP price
df['pp_lag_1'] = df['Price'].shift(1)
df['pp_lag_3'] = df['Price'].shift(3)
df['pp_lag_7'] = df['Price'].shift(7)

# Optional: time-based features
df['day_of_week'] = df['Date'].dt.dayofweek
df['month'] = df['Date'].dt.month

# Drop initial rows with NaN from lag features
df.dropna(inplace=True)

# Save final ML-ready dataset
df.to_csv("../data/processed/pp_ml_ready.csv", index=False)
print("Feature engineering complete. Saved pp_ml_ready.csv")
