import numpy as np
import pandas as pd

PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

def clean_shipments(raw, medians=None):
    df = raw.copy()
    # 1. Drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 2. Service type clean
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()
    
    # 3. Payment mode missing fill
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")
    
    # 4. Declared value clean
    val_str = df["declared_value"].astype(str).str.replace("₹", "", regex=False).str.replace(",", "", regex=False)
    df["declared_value_inr"] = pd.to_numeric(val_str, errors="coerce")
    df = df.drop(columns=["declared_value"])
    
    # 5. Dispatch time to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")
    
    # 6. Weight scanner fault sentinel (> 100kg)
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan
    
    # 7. Compute or use medians for filling missing values
    cols_to_fill = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {col: float(df[col].median()) for col in cols_to_fill}
    
    for col in cols_to_fill:
        df[col] = df[col].fillna(medians[col])
        
    return df, medians

def build_features(clean, hubs):
    df = clean.merge(hubs, on="hub_id", how="left")
    
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["log_declared_value"] = np.log(df["declared_value_inr"])
    
    return df