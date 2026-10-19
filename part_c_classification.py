import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from src.shipment_prep import FEATURES

def run_part_c():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Read modelling table
    df = pd.read_csv("data/modelling_table.csv")
    
    # 2. Define X (features) and y (target: is_delayed)
    X = df[FEATURES]
    y = df["is_delayed"]
    
    # 3. Train-test split (using fixed random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Initialize and train Classifier model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # 5. Evaluate model
    predictions = clf.predict(X_test)
    probabilities = clf.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    print(f"Classification Model Trained Successfully!")
    print(f"Test F1-Score: {f1:.4f}")
    print(f"Test ROC-AUC: {roc_auc:.4f}")
    
    # 6. Save the trained classifier model and threshold
    joblib.dump(clf, "models/delay_classifier.pkl")
    
    # Save default threshold file
    pd.DataFrame({"threshold": [0.5]}).to_csv("models/delay_threshold.csv", index=False)
    print("Classifier model and threshold saved successfully in models/")

if __name__ == "__main__":
    run_part_c()