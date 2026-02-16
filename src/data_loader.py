import pandas as pd
from typing import Tuple

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
