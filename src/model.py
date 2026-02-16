import joblib
from src.config import ModelConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from typing import Tuple
import pandas as pd
from src.config import ModelConfig


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    config: ModelConfig
) -> Tuple[Pipeline, float]:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state
    )

    model = RandomForestClassifier(random_state=config.random_state)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    joblib.dump(pipeline, config.model_path)

    return pipeline, auc
