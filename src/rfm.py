import pandas as pd

def build_rfm(df: pd.DataFrame, snapshot_date=None) -> pd.DataFrame:
    if snapshot_date is None:
        snapshot_date = df["TransactionStartTime"].max()

    rfm = (
        df.groupby("CustomerId")
        .agg(
            Recency=("TransactionStartTime",
                     lambda x: (snapshot_date - x.max()).days),
            Frequency=("TransactionId", "count"),
            Monetary=("Value", "sum")
        )
        .reset_index()
    )

    return rfm
