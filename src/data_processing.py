

import pandas as pd

def convert_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df

def build_rfm(df, customer_col, date_col, amount_col, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df[date_col].max()

    rfm = (
        df.groupby(customer_col)
        .agg(
            Recency=(date_col, lambda x: (snapshot_date - x.max()).days),
            Frequency=(customer_col, "count"),
            Monetary=(amount_col, "sum")
        )
        .reset_index()
    )
    return rfm

def cap_outliers(series, lower=0.01, upper=0.99):
    low, high = series.quantile([lower, upper])
    return series.clip(low, high)
    def preprocess_data(df):
        df=df.copy()
        