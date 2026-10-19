import os
import pandas as pd
import numpy as np
from src.shipment_prep import clean_shipments, build_features, PROMISED_HOURS, FEATURES

def run_part_a():
    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Read raw data and hubs
    raw = pd.read_csv("data/raw_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    
    # Before counts for checks
    duplicates_before = raw.duplicated().sum()
    service_types_before = raw["service_type"].nunique()
    missing_weight_before = raw["parcel_weight_kg"].loc[raw["parcel_weight_kg"] <= 100].isna().sum()
    
    # 2. Clean shipments
    clean_df, medians = clean_shipments(raw)
    service_types_after = clean_df["service_type"].nunique()
    
    # 3. Build features
    modelling_df = build_features(clean_df, hubs)
    
    # Add target: is_delayed
    modelling_df["is_delayed"] = (modelling_df["delivery_hours"] > modelling_df["promised_hours"]).astype(int)
    
    # Save modelling table
    modelling_table_path = "data/modelling_table.csv"
    modelling_df.to_csv(modelling_table_path, index=False)
    
    # Nulls in features and target count
    feature_cols_check = FEATURES + ["delivery_hours", "is_delayed"]
    nulls_count = modelling_df[feature_cols_check].isna().sum().sum()
    
    # Declared value statistics on filled column
    val_mean = round(modelling_df["declared_value_inr"].mean(), 4)
    val_median = round(modelling_df["declared_value_inr"].median(), 4)
    log_val_mean = round(modelling_df["log_declared_value"].mean(), 4)
    log_val_median = round(modelling_df["log_declared_value"].median(), 4)
    
    # Hours IQR bounds
    q25 = modelling_df["delivery_hours"].quantile(0.25)
    q75 = modelling_df["delivery_hours"].quantile(0.75)
    iqr = q75 - q25
    hours_lower = round(q25 - 1.5 * iqr, 4)
    hours_upper = round(q75 + 1.5 * iqr, 4)
    
    # Write checks output
    checks_data = [
        {"metric": "duplicates_before", "value": duplicates_before},
        {"metric": "service_type_spellings_before", "value": service_types_before},
        {"metric": "service_type_spellings_after", "value": service_types_after},
        {"metric": "missing_parcel_weight_before", "value": missing_weight_before},
        {"metric": "nulls_in_features_and_target", "value": nulls_count},
        {"metric": "declared_value_mean_inr", "value": val_mean},
        {"metric": "declared_value_median_inr", "value": val_median},
        {"metric": "log_declared_value_mean", "value": log_val_mean},
        {"metric": "log_declared_value_median", "value": log_val_median},
        {"metric": "hours_lower_bound", "value": hours_lower},
        {"metric": "hours_upper_bound", "value": hours_upper},
    ]
    pd.DataFrame(checks_data).to_csv("outputs/part_a_checks.csv", index=False)
    
    # Delay by group
    groups = []
    # Service type group
    for st, group in modelling_df.groupby("service_type"):
        groups.append({
            "dimension": "service_type", "group": st,
            "shipments": len(group), "delayed": int(group["is_delayed"].sum()),
            "delay_rate": round(group["is_delayed"].mean(), 4),
            "mean_hours": round(group["delivery_hours"].mean(), 4)
        })
    # Hub tier group
    for ht, group in modelling_df.groupby("hub_tier"):
        groups.append({
            "dimension": "hub_tier", "group": str(ht),
            "shipments": len(group), "delayed": int(group["is_delayed"].sum()),
            "delay_rate": round(group["is_delayed"].mean(), 4),
            "mean_hours": round(group["delivery_hours"].mean(), 4)
        })
    # Distance band group
    bins = [0, 200, 400, 600, 2000]
    labels = ["0-200", "201-400", "401-600", "601+"]
    modelling_df["distance_band"] = pd.cut(modelling_df["distance_km"], bins=bins, labels=labels, right=True, include_lowest=True)
    for db, group in modelling_df.groupby("distance_band", observed=False):
        groups.append({
            "dimension": "distance_band", "group": str(db),
            "shipments": len(group), "delayed": int(group["is_delayed"].sum()),
            "delay_rate": round(group["is_delayed"].mean(), 4),
            "mean_hours": round(group["delivery_hours"].mean(), 4)
        })
        
    pd.DataFrame(groups).to_csv("outputs/part_a_delay_by_group.csv", index=False)
    
    # Correlations
    numeric_cols = modelling_df.select_dtypes(include=[np.number]).columns
    corr_matrix = modelling_df[numeric_cols].corr().round(4)
    corr_matrix.to_csv("outputs/part_a_correlations.csv", index=True)
    
    print("Part A completed successfully! All files generated in data/ and outputs/")

if __name__ == "__main__":
    run_part_a()