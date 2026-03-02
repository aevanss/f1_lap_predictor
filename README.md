# F1 Lap Time Predictor 🏎️

Predicts George Russell's lap times across F1 circuits using historical race data and XGBoost.

## Overview
This project pulls real F1 telemetry and race data using the FastF1 API, builds a structured dataset across 4 seasons (2022-2025), and trains an XGBoost regression model to predict lap times per circuit.

## Key Skills Demonstrated
- REST API data collection (FastF1)
- Data cleaning and feature engineering (pandas, scikit-learn)
- Machine learning regression (XGBoost)
- Data visualisation (matplotlib)

## Results
- With sector time features: MAE of ~0.7 seconds
- Without sector times (realistic forecast): MAE of ~6 seconds
- Identified that circuit metadata (track length, corner count) would meaningfully improve predictions

## Stack
- Python, pandas, XGBoost, scikit-learn, matplotlib, FastF1

## How to Run
1. Clone the repo
2. Install dependencies: `pip install fastf1 pandas xgboost scikit-learn matplotlib`
3. Run `f1ML.py`
