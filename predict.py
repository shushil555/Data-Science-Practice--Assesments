"""
predict.py - PRT661 Life Expectancy Prediction (Sydney Group 3, Theme 2)
Model inference: loads the saved final XGBoost model and predicts life
expectancy, with a 90% prediction interval, for new country-year records.

Input: a CSV with the same columns as data/processed_data.csv
       (life expectancy itself is not needed).
Usage: python predict.py new_data.csv            -> outputs/new_predictions.csv
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

MODEL = Path("outputs/final_model.joblib")
INFO = Path("outputs/final_model_info.json")


def predict(new: pd.DataFrame) -> pd.DataFrame:
    """Return Country, Year, Status, Predicted, Lower and Upper for each row."""
    model = joblib.load(MODEL)
    info = json.load(open(INFO))
    X = new.reindex(columns=info["features"])          # same features, same order as training
    out = new[["Country", "Year", "Status"]].copy()
    out["Predicted"] = model.predict(X).astype(float).round(2)
    half = out["Status"].map(info["interval_90"]).fillna(max(info["interval_90"].values()))
    out["Lower"] = (out["Predicted"] - half).round(2)
    out["Upper"] = (out["Predicted"] + half).round(2)
    return out


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed_data.csv")
    result = predict(pd.read_csv(src))
    result.to_csv("outputs/new_predictions.csv", index=False)
    print(result.head(10).to_string(index=False))
    print(f"\nSaved {len(result)} predictions to outputs/new_predictions.csv")
