import shap
import joblib
import pandas as pd

def explain_model(data_path, model_path):
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data)
