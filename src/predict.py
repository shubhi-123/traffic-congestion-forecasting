import joblib
import pandas as pd
import numpy as np


# Load model and feature list
model = joblib.load("../models/final_xgb_model.pkl")
features = joblib.load("../models/feature_list.pkl")


def predict_traffic(data):
    """
    Predict traffic for the next 15 minutes.
    """

    df = pd.DataFrame(data)

    # Time features
    df["Hour"] = df["Hour"].astype(int)
    df["Minute"] = df["Minute"].astype(int)

    # Weekday
    df["Weekday"] = df["Weekday"].astype(int)

    # Rush hour: 6-9 AM and 4-7 PM
    df["Rush_Hour"] = df["Hour"].apply(
        lambda x: 1 if (6 <= x <= 9) or (16 <= x <= 19) else 0
    )

    # Lag features
    df["Total_lag1"] = df["Total"].shift(1)
    df["Total_lag2"] = df["Total"].shift(2)
    df["Total_lag4"] = df["Total"].shift(4)
    df["Total_lag96"] = df["Total"].shift(96)

    # Rolling features
    df["Rolling_Mean_4"] = (
        df["Total"]
        .shift(1)
        .rolling(window=4)
        .mean()
    )

    df["Rolling_Mean_8"] = (
        df["Total"]
        .shift(1)
        .rolling(window=8)
        .mean()
    )

    df["Rolling_Std_4"] = (
        df["Total"]
        .shift(1)
        .rolling(window=4)
        .std()
    )

    # Use the latest row
    latest = df.iloc[-1:]

    # Make prediction
    prediction = model.predict(latest[features])

    return float(prediction[0])