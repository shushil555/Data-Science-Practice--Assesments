"""
data_pipeline.py - PRT661 Life Expectancy Prediction (Sydney Group 3, Theme 2)
Cleans the WHO Life Expectancy dataset and saves a model-ready file.

Missing predictor values are NOT filled here. Imputation happens inside the
model pipeline, fitted on training data only, so no test information leaks
into training (same principle as Assessment 2).

Usage:  python data_pipeline.py ["Life Expectancy Data.csv"]
Output: data/processed_data.csv  (upload to s3://prt661-life-expectancy/processed/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Life Expectancy Data.csv")
OUT = Path("data/processed_data.csv")
TARGET = "Life expectancy"


def main():
    # 1. Load and tidy column names (raw file has stray spaces, e.g. "Life expectancy ")
    df = pd.read_csv(RAW)
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns, "
          f"{df['Country'].nunique()} countries")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # 2. Drop rows with no target value (cannot be used for supervised learning)
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print(f"Removed {before - len(df)} rows with missing target -> {len(df)} rows")

    # 3. Log transform right-skewed predictors (planned in A2)
    df["log_GDP"] = np.log1p(df["GDP"])
    df["log_Population"] = np.log1p(df["Population"])

    # 4. Encode Status (Developed = 1, Developing = 0)
    df["Status_Developed"] = (df["Status"] == "Developed").astype(int)

    # 5. Report remaining missingness (handled later by the model pipeline)
    miss = df.isna().sum()
    print("Missing values still present (imputed at training time):")
    print(miss[miss > 0].sort_values(ascending=False).to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Pipeline complete. Saved {df.shape[0]} rows to {OUT}")


if __name__ == "__main__":
    main()
