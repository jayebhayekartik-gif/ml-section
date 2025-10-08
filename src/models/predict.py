import pandas as pd
import joblib
import os

def generate_predictions(
    input_csv='D:/shantanu/git/ml-section/data/processed/processed_pp_prices_clean.csv',
    model_path='D:/shantanu/git/ml-section/models/linear_model.pkl',
    output_csv='D:/shantanu/git/ml-section/data/processed/predictions.csv'
):
    if not os.path.exists(input_csv):
        print(f"❌ Input CSV not found: {input_csv}")
        return
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    df = pd.read_csv(input_csv)
    
    # Create numeric feature from Date
    df['Day'] = pd.to_datetime(df['Date']).dt.dayofyear
    
    model = joblib.load(model_path)

    df['predicted_price'] = model.predict(df[['Day']])
    df.to_csv(output_csv, index=False)
    print(f"✅ Predictions saved to {output_csv}")

if __name__ == "__main__":
    generate_predictions()

