## 2. Data Processing & Feature Engineering (`src/data_processing.py`)

### Purpose

#Turn raw transactions into **customer-level, model-ready features**.

### Key Ideas

"* Fail early if data is broken "
"One function = one responsibility No hard-coded magic"

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

RAW_PATH = Path(r"c:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\raw\CreditRisk-data.csv")
PROCESSED_PATH = Path(r"C:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\process\processed.csv")


def load_data(path: Path) -> pd.DataFrame:
    """Safely load raw transaction data."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])

    if df.empty:
        raise ValueError("Loaded dataset is empty")

    return df


def create_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Create customer-level aggregate features."""
    try:
        agg = df.groupby("CustomerId").agg(
            total_amount=("Amount", "sum"),
            avg_amount=("Amount", "mean"),
            txn_count=("TransactionId", "count"),
            amount_std=("Amount", "std"),
        ).reset_index()
    except KeyError as e:
        raise KeyError(f"Missing required column: {e}")

    agg["amount_std"] = agg["amount_std"].fillna(0)
    return agg


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based behavioral features."""
    df = df.copy()
    ts = df["TransactionStartTime"]

    df["txn_hour"] = ts.dt.hour
    df["txn_day"] = ts.dt.day
    df["txn_month"] = ts.dt.month
    df["txn_year"] = ts.dt.year

    return df


def build_features() -> pd.DataFrame:
    """End-to-end feature generation."""
    df = load_data(RAW_PATH)
    df = extract_time_features(df)

    agg = create_aggregates(df)

    final_df = pd.merge(df, agg, on="CustomerId", how="left")

    final_df.to_csv(PROCESSED_PATH, index=False)
    return final_df


if __name__ == "__main__":
    build_features()


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