import pandas as pd

def cap_outliers(df: pd.DataFrame, column: str, q=0.99) -> pd.DataFrame:
    cap = df[column].quantile(q)
    df[column] = df[column].clip(upper=cap)
    return df
