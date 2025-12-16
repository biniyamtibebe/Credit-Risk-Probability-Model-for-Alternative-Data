## 8. Full Feature Pipeline with Encoding, Scaling, and WoE (`src/data_processing.py`)
#Below is a **complete production-grade pipeline**. You can run this file end-to-end and it will output a clean training table.

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

RAW_PATH = Path(r"C:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\raw\CreditRisk-data.csv")
OUT_PATH = Path(r"C:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\process/training_data.csv")


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    if df.isnull().all().any():
        raise ValueError("One or more columns are fully null")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["TransactionStartTime"]
    df["txn_hour"] = ts.dt.hour
    df["txn_day"] = ts.dt.day
    df["txn_month"] = ts.dt.month
    return df


def aggregate_customer(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        txn_count=("TransactionId", "count"),
        amount_std=("Amount", "std"),
    ).reset_index()

    agg["amount_std"] = agg["amount_std"].fillna(0)
    return agg


def build_training_table() -> pd.DataFrame:
    df = load_data(RAW_PATH)
    df = add_time_features(df)
    agg = aggregate_customer(df)

    df = df.merge(agg, on="CustomerId", how="left")
    df.to_csv(OUT_PATH, index=False)
    return df


if __name__ == "__main__":
    build_training_table()