import shap
import matplotlib.pyplot as plt
import pandas as pd

def compute_shap(model, X_sample: pd.DataFrame):
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    X_transformed = model.named_steps["preprocessor"].transform(X_sample)
    shap_values = explainer.shap_values(X_transformed)
    return explainer, shap_values

def plot_global_shap(explainer, shap_values, X_transformed):
    shap.summary_plot(shap_values, X_transformed)
