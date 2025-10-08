import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def train_linear_regression(
    data_path='D:/shantanu/git/ml-section/data/processed/processed_pp_prices_clean.csv', 
    model_path='linear_model.pkl'
):
    # Load dataset
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Convert Date to numeric feature
    df['Day'] = pd.to_datetime(df['Date']).dt.dayofyear

    X = df[['Day']]  # features
    y = df['Price']  # target

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_linear_regression()

# src/models/train.py
from xgboost import XGBRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_xgboost(X, y, config):
    """
    Train XGBoost model
    X: features (DataFrame or np.array)
    y: target (Series or np.array)
    config: dict with parameters like n_estimators, learning_rate
    """
    model = XGBRegressor(
        n_estimators=config.get("n_estimators", 500),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 3),
        random_state=config.get("random_state", 42)
    )
    model.fit(X, y)
    return model

def train_sarimax(X, y, config):
    """
    Train SARIMAX model
    X: features (usually time series index)
    y: target
    config: dict with SARIMAX order parameters
    """
    # For SARIMAX, we usually only pass the target (univariate)
    order = config.get("order", (1,1,1))
    seasonal_order = config.get("seasonal_order", (0,0,0,0))
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit(disp=False)
    y_pred = model_fit.predict(start=0, end=len(y)-1)
    return model_fit, y_pred





