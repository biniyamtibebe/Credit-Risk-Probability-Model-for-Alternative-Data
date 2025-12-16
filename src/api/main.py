import mlflow
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")  # Adjust if your MLflow server runs on a different host or port
# Start a new MLflow run
with mlflow.start_run() as run:
    # Log parameters and metrics
    mlflow.log_param("param1", 5)
    mlflow.log_metric("metric1", np.random.random())
    
    # Create and train a model
    X, y = make_classification(n_samples=100, n_features=10)
    model = LogisticRegression()
    model.fit(X, y)
    
    # Log the model
    mlflow.sklearn.log_model(model, "model")  # Ensure this matches your loading path
    print("Logged run_id:", run.info.run_id)
    
    # Optionally register the model
    mlflow.register_model(f"runs:/{run.info.run_id}/model", "CreditRiskModel")

# Specify your experiment name
experiment_name = "CreditRiskModel"
# Check if the experiment exists, if not, create it
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    print(f"Experiment '{experiment_name}' does not exist. Creating a new one.")
    experiment_id = mlflow.create_experiment(experiment_name)
else:
    experiment_id = experiment.experiment_id

# Search for runs in that experiment
runs = mlflow.search_runs(experiment_ids=[experiment_id])
print(runs[['run_id', 'status']])

# Load the registered model
try:
    model = mlflow.sklearn.load_model("models:/CreditRiskModel/latest")
except mlflow.exceptions.MlflowException as e:
    print(f"Failed to load model: {e}")

# Get details about a specific run
run_data = mlflow.get_run(run.info.run_id)
print(run_data.data)

from fastapi import FastAPI, HTTPException
import mlflow.sklearn
import pandas as pd

app = FastAPI()

# Load the model from the registered model
model = mlflow.sklearn.load_model("models:/CreditRiskModel/latest")

def probability_to_score(probability):
    """Convert probability to a score out of 100."""
    return int(probability * 100)

def recommend_loan(score):
    """Provide loan recommendation based on the credit score."""
    return "Recommended" if score > 50 else "Not Recommended"

@app.post("/predict")
def predict(features: dict):
    try:
        # Convert the input features to a DataFrame
        df = pd.DataFrame([features])
        
        # Get the probability of default
        prob = model.predict_proba(df)[0, 1]
        
        # Convert probability to credit score
        score = probability_to_score(prob)
        
        # Get loan recommendation
        loan = recommend_loan(score)

        return {
            "risk_probability": prob,
            "credit_score": score,
            "loan_offer": loan
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))