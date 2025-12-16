## 4. Model Training with MLflow (`src/train.py`)
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import pandas as pd


def train_models(df: pd.DataFrame):
    X = df.drop(columns=["CustomerId", "is_high_risk"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2,
        random_state=42, stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "GradientBoosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            auc = roc_auc_score(y_test, probs)

            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(model, "model")

            print(f"{name} AUC: {auc:.4f}")


## 10. Model Training with Multiple Models (`src/train.py`)
#This trains **Logistic Regression and Gradient Boosting**, logs both, and selects the best.

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


def train_and_log(X, y, model, name):
    with mlflow.start_run(run_name=name):
        model.fit(X, y)
        preds = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, preds)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(model, "model")
        return auc


def main(df: pd.DataFrame):
    X = df.drop(columns=["is_high_risk"])
    y = df["is_high_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    lr = LogisticRegression(max_iter=1000)
    gb = GradientBoostingClassifier(random_state=42)

    auc_lr = train_and_log(X_train, y_train, lr, "LogisticRegression")
    auc_gb = train_and_log(X_train, y_train, gb, "GradientBoosting")

    print(f"LR AUC: {auc_lr:.3f}, GB AUC: {auc_gb:.3f}")