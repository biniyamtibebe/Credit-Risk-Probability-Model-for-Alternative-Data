import pandas as pd
from src.config import ModelConfig
from src.preprocessing import build_preprocessor
from src.model import train_model

def test_model_training():
    df = pd.DataFrame({
        "age": [25, 35, 45, 50],
        "income": [20000, 40000, 60000, 80000],
        "default": [0, 0, 1, 1]
    })

    X = df[["age", "income"]]
    y = df["default"]

    preprocessor = build_preprocessor(["age", "income"], [])
    config = ModelConfig()

    model, auc = train_model(X, y, preprocessor, config)

    assert 0 <= auc <= 1
