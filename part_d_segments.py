import os
import pandas as pd
import numpy as np
import joblib
from src.shipment_prep import clean_shipments, build_features, FEATURES

def run_part_d():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # 1. Read data and trained models
    modelling_df = pd.read_csv("data/modelling_table.csv")
    hubs = pd.read_csv("data/hubs.csv")
    new_raw = pd.read_csv("data/new_shipments.csv")
    
    # Load models
    clf = joblib.load("models/delay_classifier.pkl")
    reg = joblib.load("models/delay_model.pkl")
    
    # 2. Process new shipments using cleaning and feature functions
    clean_new, _ = clean_shipments(new_raw)
    new_features_df = build_features(clean_new, hubs)
    
    # Ensure exact feature order
    X_new = new_features_df[FEATURES]
    
    # 3. Predict delay probabilities and predicted hours for new shipments
    new_features_df["predicted_delay_probability"] = clf.predict_proba(X_new)[:, 1]
    new_features_df["predicted_delivery_hours"] = reg.predict(X_new)
    
    # 4. Generate action list based on risk probability
    # Action rule: High Risk (>0.5 probability) -> 'Proactive Call / Escalate', Else -> 'Standard Dispatch'
    conditions = [
        new_features_df["predicted_delay_probability"] > 0.5
    ]
    choices = ["Proactive Call / Escalate"]
    new_features_df["action_recommendation"] = np.select(conditions, choices, default="Standard Dispatch")
    
    # Select output columns for action list
    action_columns = ["shipment_id", "hub_id", "service_type", "predicted_delay_probability", "predicted_delivery_hours", "action_recommendation"]
    action_list = new_features_df[action_columns]
    
    # Save outputs
    action_list.to_csv("outputs/part_d_action_list.csv", index=False)
    
    print("Part D completed successfully! Action list generated in outputs/part_d_action_list.csv")

if __name__ == "__main__":
    run_part_d()