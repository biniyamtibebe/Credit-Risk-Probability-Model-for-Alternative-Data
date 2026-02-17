import os
import joblib
from typing import Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config import ModelConfig


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    config: ModelConfig
) -> Tuple[Pipeline, float]:

    # Safe stratification:
    # Only stratify if dataset is large enough and has more than 1 class
    if len(y) >= 10 and y.nunique() > 1:
        stratify = y
    else:
        stratify = None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify
    )

    # Model (handle class imbalance)
    model = RandomForestClassifier(
        random_state=config.random_state,
        class_weight="balanced"
    )

    # Full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    # Ensure model directory exists before saving
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    # Save trained pipeline
    joblib.dump(pipeline, config.model_path)

    return pipeline, auc
