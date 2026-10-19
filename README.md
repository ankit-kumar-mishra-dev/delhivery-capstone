# Delhivery Last-Mile Delay Intelligence Pipeline

Welcome to my Capstone Project repository for the AI & ML Certification Program. This project builds an end-to-end analytics and machine learning pipeline for Delhivery's regional operations to predict delivery hours, flag potential delays proactively, and evaluate hub performance.

# Project Structure

The repository is organized as follows:

<repo>/
├── README.md
├── requirements.txt
├── generate_data.py
├── data/
│   ├── hubs.csv
│   ├── raw_shipments.csv
│   ├── new_shipments.csv
│   └── modelling_table.csv
├── src/
│   └── shipment_prep.py
├── part_a_clean_eda.py
├── part_b_regression.py
├── part_c_classification.py
├── part_d_segments.py
├── models/
│   ├── delay_model.pkl
│   ├── delay_classifier.pkl
│   ├── delay_threshold.csv
│   └── feature_order.csv
└── outputs/
    ├── part_a_checks.csv
    ├── part_a_delay_by_group.csv
    ├── part_a_correlations.csv
    └── part_d_action_list.csv

# Reproducibility and Execution Steps

To run this project from scratch on a clean clone, execute the scripts in the following order using terminal commands:

1. Install Dependencies:
   pip install -r requirements.txt

2. Generate Synthetic Data:
   python generate_data.py

3. Run Part A (Data Cleaning and EDA):
   python part_a_clean_eda.py

4. Run Part B (Regression Model for Delivery Hours):
   python part_b_regression.py

5. Run Part C (Classification Model for Delay Detection):
   python part_c_classification.py

6. Run Part D (Hub Segments and Action List Generation):
   python part_d_segments.py

# Data Findings and Insights

- Service Type Delays: Standard shipments experience a notable delay share compared to Express deliveries due to longer transit windows and distance dependencies.
- Hub Tiers: Performance varies across Tier 1, Tier 2, and Tier 3 hubs, reflecting operational load and resource distribution.
- Distance Impact: Longer transit distance bands directly correlate with higher mean delivery hours.
- Log Transformation: declared_value_inr was log-transformed to normalize skewed financial distributions and stabilize variance for machine learning models.

# Model Performance Summary

- Regression Model (Part B): Trained using a Random Forest Regressor to predict delivery_hours accurately based on shipment and hub features.
- Classification Model (Part C): Trained using a Random Forest Classifier to detect delays, achieving robust F1-score and ROC-AUC metrics.
- Operational Action List (Part D): Automatically generates proactive escalation flags for high-risk shipments before dispatch deadlines.
-