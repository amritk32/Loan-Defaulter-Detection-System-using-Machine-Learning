from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
from _pydantic import LoanApplication

 
class PredictionRequest(LoanApplication):
    selected_threshold: float = 0.5


app = FastAPI(title="Loan Default Prediction API")


def load_artifacts():
    preprocessor_payload = joblib.load("preprocessor.pkl")
    model_payload = joblib.load("best_model.pkl")
    threshold_metrics = pd.read_csv("threshold_metrics.csv")
    template = pd.read_csv("default_template.csv")

    return {
        "preprocessor": preprocessor_payload["preprocessor"],
        "model": model_payload["model"],
        "model_name": model_payload.get("model_name", "Unknown Model"),
        "threshold_metrics": threshold_metrics,
        "template": template,
    }


artifacts = load_artifacts()
preprocessor = artifacts["preprocessor"]
model = artifacts["model"]
model_name = artifacts["model_name"]
threshold_metrics_df = artifacts["threshold_metrics"]
template = artifacts["template"]


@app.get("/")
def root():
    return {"message": "Loan defaulter detection system running 🚀"}


@app.post("/predict")
def predict(request: PredictionRequest):
    input_df = template.copy()

    input_df["annual_inc"] = request.annual_inc
    input_df["loan_amnt"] = request.loan_amnt
    input_df["int_rate"] = request.int_rate
    input_df["emp_length"] = request.emp_length
    input_df["dti"] = request.dti
    input_df["term"] = request.term
    input_df["home_ownership"] = request.home_ownership
    input_df["purpose"] = request.purpose
    input_df["total_acc"] = request.total_acc

    transformed_df = preprocessor.transform(input_df)

    if hasattr(model, "predict_proba"):
        prediction_prob = float(model.predict_proba(transformed_df)[0][1])
    else:
        decision = float(model.decision_function(transformed_df)[0])
        prediction_prob = 1 / (1 + np.exp(-decision))

    threshold_used = float(request.selected_threshold)
    is_risky = bool(prediction_prob >= threshold_used)

    threshold_row = threshold_metrics_df[threshold_metrics_df["threshold"] == threshold_used]
    if len(threshold_row) > 0:
        threshold_precision = float(threshold_row.iloc[0]["precision"])
        threshold_recall = float(threshold_row.iloc[0]["recall"])
        threshold_f1 = float(threshold_row.iloc[0]["f1_score"])
    else:
        threshold_precision = None
        threshold_recall = None
        threshold_f1 = None

    return {
        "predicted_output": "unsafe" if is_risky else "safe",
        "risk_probability": round(prediction_prob, 4),
        "threshold_used": threshold_used,
        "model_name": model_name,
        "precision": threshold_precision,
        "recall": threshold_recall,
        "f1_score": threshold_f1,
        "threshold_metrics": threshold_metrics_df.round(4).to_dict(orient="records"),
    }
