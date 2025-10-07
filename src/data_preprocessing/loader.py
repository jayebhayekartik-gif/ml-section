# src/data/loader.py

import pandas as pd
import logging

def load_data(file_path):
    """
    Load raw CSV data.
    """
    logging.info(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise
