from src.data_loader import load_data, split_features_target
from src.preprocessing import build_preprocessor
from src.model import train_model
from src.config import ModelConfig


if __name__ == "__main__":

    config = ModelConfig()

    df = load_data("data/raw/CreditRisk-data.csv")

    print("Available columns:", df.columns.tolist())

    X, y = split_features_target(df, target=config.target_column)

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    model, auc = train_model(X, y, preprocessor, config)

    print(f"Model trained successfully. ROC-AUC: {auc:.4f}")

