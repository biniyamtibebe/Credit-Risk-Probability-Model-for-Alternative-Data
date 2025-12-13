# src/data_processing.py
import pandas as pd
from pathlib import Path

RAW = Path(r"c:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\raw\CreditRisk-data.csv")


def load_transactions(path: str):
    df = pd.read_csv(path, parse_dates=["TransactionStartTime"])
    # Basic validation
    expected = {"TransactionId","AccountId","CustomerId","Amount","Value","TransactionStartTime"}
    assert expected.issubset(set(df.columns)), "Missing columns"
    return df

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # basic cleaning
    df = df.drop_duplicates(subset=["TransactionId"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    # fill na, filter bad rows
    # …
    return df

def save_customer_aggregates(df, out=r"c:\Users\hp\Pictures\Bati Bank\Credit-Risk-Probability-Model-for-Alternative-Data\data\process\customers.csv"):
    df.to_parquet(out)

