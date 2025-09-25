import pandas as pd
import os

# Load raw CSVs
pp = pd.read_csv(r"D:\shantanu\git\ml-section\data\raw\pp_prices.csv")
naphtha = pd.read_csv(r"D:\shantanu\git\ml-section\data\raw\naphtha_prices.csv")
propylene = pd.read_csv(r"D:\shantanu\git\ml-section\data\raw\propylene_prices.csv")
freight = pd.read_csv(r"D:\shantanu\git\ml-section\data\raw\freight_cost.csv")

# Convert Date to datetime and sort
for df in [pp, naphtha, propylene, freight]:
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)

# Fill missing values using forward fill (avoiding FutureWarning)
for df in [pp, naphtha, propylene, freight]:
    df['Price'] = df['Price'].ffill()

# Create processed folder if not exists
os.makedirs("../data/processed", exist_ok=True)

# Save cleaned files to processed folder
pp.to_csv("../data/processed/processed_pp_prices_clean.csv", index=False)
naphtha.to_csv("../data/processed/processed_naphtha_prices_clean.csv", index=False)
propylene.to_csv("../data/processed/processed_propylene_prices_clean.csv", index=False)
freight.to_csv("../data/processed/processed_freight_cost_clean.csv", index=False)

print("Data preprocessing complete. Cleaned files saved in 'processed/' folder.")
