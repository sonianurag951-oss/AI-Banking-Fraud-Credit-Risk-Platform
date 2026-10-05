from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predict loan default risk using ML models",
    version="1.0.0"
)

# LOAD MODEL AND SCALER
# Get the directory where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(BASE_DIR, "Models")


# Load best model (XGBoost or Logistic Regression)
try:
    model = joblib.load(os.path.join(models_dir, "credit_risk_xgb.pkl"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    scaler = None

feature_cols = [
    'loan_amount', 'loan_term', 'interest_rate', 'monthly_income',
    'existing_debt', 'previous_defaults', 'late_payments',
    'age', 'income', 'credit_score', 'account_tenure', 'dependents',
    'debt_to_income', 'loan_to_income', 'payment_burden', 'risk_score',
    'employment_type_encoded'
]

# REQUEST MODEL 
from pydantic import BaseModel, ConfigDict

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
                "income": 75000,
                "credit_score": 720,
                "account_tenure": 5,
                "dependents": 2,
                "employment_type": "Salaried"
            }
        }
    )
    
    loan_amount: float
    loan_term: int
    interest_rate: float
    monthly_income: float
    existing_debt: float
    previous_defaults: int
    late_payments: int
    age: int
    income: float
    credit_score: int
    account_tenure: int
    dependents: int
    employment_type: str


def encode_employment_type(employment_type: str) -> int:
    """Encode employment type to numeric value"""
    mapping = {
        'Salaried': 0,
        'Self-employed': 1,
        'Student': 2,
        'Unemployed': 3,
        'Retired': 4
    }
    return mapping.get(employment_type, 0)

def calculate_features(data: dict) -> dict:
    """Calculate engineered features"""
    data['debt_to_income'] = data['existing_debt'] / data['monthly_income']
    data['loan_to_income'] = data['loan_amount'] / (data['monthly_income'] * 12)
    data['payment_burden'] = data['loan_amount'] / data['loan_term']
    data['risk_score'] = (
        0.35 * (850 - data['credit_score']) / 550
        + 0.30 * data['debt_to_income']
        + 0.20 * data['previous_defaults']
        + 0.15 * data['late_payments']
    )
    data['employment_type_encoded'] = encode_employment_type(data['employment_type'])
    return data

def predict_risk(features: list) -> dict:
    """Make prediction and return risk assessment"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    # Risk category
    if probability < 0.3:
        risk_category = "Low Risk"
        recommendation = "Approve loan"
    elif probability < 0.6:
        risk_category = "Medium Risk"
        recommendation = "Review manually"
    else:
        risk_category = "High Risk"
        recommendation = "Reject loan"
    
    return {
        "prediction": int(prediction),
        "default_probability": round(float(probability), 4),
        "risk_category": risk_category,
        "recommendation": recommendation
    }

#  API ENDPOINTS

@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "model": "Credit Risk Prediction",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    """Check if model is loaded"""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
def predict_loan_default(application: LoanApplication):
    """
    Predict loan default risk for a loan application
    
    Returns:
    - prediction: 0 (No Default) or 1 (Default)
    - default_probability: Probability of default (0-1)
    - risk_category: Low/Medium/High Risk
    - recommendation: Approve/Review/Reject
    """
    try:
        # Convert to dict
        data = application.model_dump()
        
        # Calculate engineered features
        data = calculate_features(data)
        
        # Extract features in correct order
        features = [data[col] for col in feature_cols]
        
        # Make prediction
        result = predict_risk(features)
        
        # Add input data to response
        result["input_data"] = data
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch_loans(applications: list[LoanApplication]):
    """
    Predict loan default risk for multiple loan applications
    """
    try:
        results = []
        for application in applications:
            data = application.model_dump()
            data = calculate_features(data)
            features = [data[col] for col in feature_cols]
            result = predict_risk(features)
            result["input_data"] = data
            results.append(result)
        
        return {
            "total_predictions": len(results),
            "predictions": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature-importance")
def get_feature_importance():
    """
    Get feature importance from the model (if available)
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return {
                "feature_importance": importance_df.to_dict('records')
            }
        else:
            return {"message": "Feature importance not available for this model"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

