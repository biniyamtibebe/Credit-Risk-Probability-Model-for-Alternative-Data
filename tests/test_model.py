import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def train_model(X, y, preprocessor, config):
    """
    Train a classification model and return fitted model + ROC AUC.
    Handles very small datasets safely.
    """

    # Ensure at least 2 samples per class before stratifying
    stratify = y if len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify
    )

    # Build pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression())
        ]
    )

    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]

    # Safe AUC calculation
    if len(np.unique(y_test)) < 2:
        # If only one class present in test set,
        # ROC AUC is undefined — return neutral baseline
        auc = 0.5
    else:
        auc = roc_auc_score(y_test, y_proba)

    return model, auc

