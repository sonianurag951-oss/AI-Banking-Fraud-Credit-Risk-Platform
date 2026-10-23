import os
from typing import Literal

import joblib
import keras
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict loan default risk using an ANN model",
    version="1.1.0",
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "Models")

MODEL_PATH = os.path.join(models_dir, "credit_risk_ann.keras")
SCALER_PATH = os.path.join(models_dir, "scaler.pkl")

model = None
scaler = None

try:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ANN model and scaler loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")

feature_cols = [
    "loan_amount", "loan_term", "interest_rate", "monthly_income",
    "existing_debt", "previous_defaults", "late_payments",
    "age", "income", "credit_score", "account_tenure", "dependents",
    "debt_to_income", "loan_to_income", "payment_burden", "risk_score",
    "employment_type_encoded",
]

LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6
DECISION_THRESHOLD = 0.5  

# Request model
class LoanApplication(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "loan_amount": 500000,
                "loan_term": 36,
                "interest_rate": 12.5,
                "monthly_income": 75000,
                "existing_debt": 150000,
                "previous_defaults": 0,
                "late_payments": 2,
                "age": 32,
                "income": 900000,
                "credit_score": 720,
                "account_tenure": 5,
                "dependents": 2,
                "employment_type": "Salaried",
            }
        }
    )

    loan_amount: float = Field(gt=0)
    loan_term: int = Field(gt=0, le=480)
    interest_rate: float = Field(ge=0, le=100)
    monthly_income: float = Field(gt=0)
    existing_debt: float = Field(ge=0)
    previous_defaults: int = Field(ge=0)
    late_payments: int = Field(ge=0)
    age: int = Field(ge=18, le=100)
    income: float = Field(gt=0)
    credit_score: int = Field(ge=300, le=850)
    account_tenure: int = Field(ge=0)
    dependents: int = Field(ge=0)
    employment_type: Literal[
        "Salaried", "Self-employed", "Student", "Unemployed", "Retired"
    ]


# Feature engineering
EMPLOYMENT_MAP = {
    "Salaried": 0,
    "Self-employed": 1,
    "Student": 2,
    "Unemployed": 3,
    "Retired": 4,
}


def calculate_features(data: dict) -> dict:
    data["debt_to_income"] = data["existing_debt"] / data["monthly_income"]
    data["loan_to_income"] = data["loan_amount"] / (data["monthly_income"] * 12)
    data["payment_burden"] = data["loan_amount"] / data["loan_term"]
    data["risk_score"] = (
        0.35 * (850 - data["credit_score"]) / 550
        + 0.30 * data["debt_to_income"]
        + 0.20 * data["previous_defaults"]
        + 0.15 * data["late_payments"]
    )
    data["employment_type_encoded"] = EMPLOYMENT_MAP[data["employment_type"]]
    return data


# Prediction
def risk_from_probability(probability: float) -> tuple[str, str]:
    if probability < LOW_RISK_THRESHOLD:
        return "Low Risk", "Approve loan"
    if probability < HIGH_RISK_THRESHOLD:
        return "Medium Risk", "Review manually"
    return "High Risk", "Reject loan"


def score_applications(applications: list[LoanApplication]) -> list[dict]:
    """Score one or many applications with a single model call."""
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    enriched = []
    rows = []
    for application in applications:
        data = calculate_features(application.model_dump())
        enriched.append(data)
        rows.append([data[col] for col in feature_cols])

    # DataFrame
    X = pd.DataFrame(rows, columns=feature_cols)
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled, verbose=0).flatten()

    results = []
    for data, probability in zip(enriched, probabilities):
        probability = float(probability)
        category, recommendation = risk_from_probability(probability)
        results.append({
            "prediction": int(probability >= DECISION_THRESHOLD),
            "default_probability": round(probability, 4),
            "risk_category": category,
            "recommendation": recommendation,
            "input_data": data,
        })
    return results

# endpoint
@app.get("/")
def read_root():
    return {
        "status": "online",
        "model": "Credit Risk Prediction (ANN)",
        "version": "1.1.0",
    }


@app.get("/health")
def health_check():
    loaded = model is not None and scaler is not None
    return {
        "status": "healthy" if loaded else "unhealthy",
        "model_loaded": loaded,
    }


@app.post("/predict")
def predict_loan_default(application: LoanApplication):
    """Predict default risk for one loan application."""
    try:
        return score_applications([application])[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-batch")
def predict_batch_loans(applications: list[LoanApplication]):
    """Predict default risk for multiple loan applications."""
    try:
        results = score_applications(applications)
        return {"total_predictions": len(results), "predictions": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-importance")
def get_feature_importance():
    """ANN has no built-in feature importance."""
    return {
        "message": (
            "Feature importance is not available for the ANN model. "
            "See the SHAP plots saved in the Results folder."
        )
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)