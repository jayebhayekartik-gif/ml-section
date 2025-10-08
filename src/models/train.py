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




