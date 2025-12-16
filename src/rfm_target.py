## 9. RFM Target Creation Script (`src/rfm_target.py`)
#This script **creates the proxy default label** and persists it.

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot - x.max()).days),
        frequency=("TransactionId", "count"),
        monetary=("Amount", "sum"),
    ).reset_index()

    return rfm


def label_high_risk(rfm: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm["cluster"] = kmeans.fit_predict(X)

    summary = rfm.groupby("cluster").mean()
    high_risk_cluster = summary["frequency"].idxmin()

    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)
    return rfm[["CustomerId", "is_high_risk"]]