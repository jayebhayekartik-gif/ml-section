# preprocess.py
# src/data/preprocess.py

import pandas as pd
import logging

def preprocess_data(df):
    """
    Basic preprocessing:
    - Fill missing values
    - Convert dates
    """
    logging.info("Starting preprocessing...")
    
    # Example: fill numeric NaNs with median
    numeric_cols = df.select_dtypes(include=['float64','int64']).columns
    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)
    
    # Example: convert date column if exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
    
    logging.info("Preprocessing completed.")
    return df

