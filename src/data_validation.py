import pandas as pd

REQUIRED_COLUMNS = {
    "TransactionId",
    "AccountId",
    "CustomerId",
    "Amount",
    "TransactionStartTime",
    "CountryCode",
    "CurrencyCode"
}

def validate_schema(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
