import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Datetime ---
    df["TransactionStartTime"] = pd.to_datetime(
        df["TransactionStartTime"], utc=True, errors="coerce"
    )

    # --- Categorical casting ---
    df["CountryCode"] = df["CountryCode"].astype("category")
    df["CurrencyCode"] = df["CurrencyCode"].astype("category")

    # --- Amount handling ---
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
    df["Value"] = df["Amount"].abs()

    # --- Missing handling ---
    df = df.dropna(subset=["TransactionStartTime", "Amount"])

    return df
