import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.shipment_prep import FEATURES

def run_part_b():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Read modelling table
    df = pd.read_csv("data/modelling_table.csv")
    
    # 2. Define X (features) and y (target)
    X = df[FEATURES]
    y = df["delivery_hours"]
    
    # 3. Train-test split (using a fixed random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and train Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Regression Model Trained Successfully!")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # 6. Save the trained model
    model_path = "models/delay_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save feature order for safety during prediction
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)

if __name__ == "__main__":
    run_part_b()