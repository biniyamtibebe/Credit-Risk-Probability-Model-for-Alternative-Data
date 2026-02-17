import pandas as pd


def load_data(path: str):
    """
    Load dataset from CSV file.
    """
    df = pd.read_csv(path)
    return df


def split_features_target(df, target: str):
    """
    Split dataframe into features (X) and target (y).
    """
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target])
    y = df[target]
    return X, y
