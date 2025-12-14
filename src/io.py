from pathlib import Path

def save_processed(df, path="data/processed/clean_data.csv"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
