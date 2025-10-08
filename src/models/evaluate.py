import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

def evaluate_model(
    data_path='D:/shantanu/git/ml-section/data/processed/processed_pp_prices_clean.csv',
    model_path='D:/shantanu/git/ml-section/models/linear_model.pkl'
):
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    df = pd.read_csv(data_path)
    
    # Convert Date to numeric feature
    df['Day'] = pd.to_datetime(df['Date']).dt.dayofyear
    X = df[['Day']].values
    y = df['Price'].values

    model = joblib.load(model_path)
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"✅ Mean Squared Error: {mse:.2f}")

if __name__ == "__main__":
    evaluate_model()
