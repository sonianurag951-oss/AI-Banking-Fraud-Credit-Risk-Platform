import os
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  
warnings.filterwarnings("ignore")

import joblib
from tensorflow import keras
import pandas as pd
import streamlit as st

from chatbot import render_chatbot

st.set_page_config(
    page_title="Credit Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .risk-low {
        background-color: #15803D;
        color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #111111;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #111111;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .risk-low h2, .risk-low h3, .risk-low p,
    .risk-medium h2, .risk-medium h3, .risk-medium p,
    .risk-high h2, .risk-high h3, .risk-high p {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)

FEATURE_COLS = [
    "loan_amount", "loan_term", "interest_rate", "monthly_income",
    "existing_debt", "previous_defaults", "late_payments",
    "age", "income", "credit_score", "account_tenure", "dependents",
    "debt_to_income", "loan_to_income", "payment_burden", "risk_score",
    "employment_type_encoded",
]

EMPLOYMENT_MAP = {
    "Salaried": 0,
    "Self-employed": 1,
    "Student": 2,
    "Unemployed": 3,
    "Retired": 4,
}

LOW_RISK_THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6
DECISION_THRESHOLD = 0.5  # same threshold used for the ANN in project.py
MODEL_FILE = "credit_risk_ann.keras"


# Model loading
def find_models_dir():
    """Look for the Models folder next to this file, one level up, or in cwd."""
    here = Path(__file__).resolve().parent
    candidates = []
    if os.getenv("MODELS_DIR"):
        candidates.append(Path(os.environ["MODELS_DIR"]))
    candidates += [here / "Models", here.parent / "Models", Path.cwd() / "Models"]

    for folder in candidates:
        if (folder / MODEL_FILE).exists():
            return folder
    return None


@st.cache_resource
def load_models():
    """Load ANN + scaler once (cached). Returns (model, scaler, error_text)."""
    models_dir = find_models_dir()
    if models_dir is None:
        return None, None, f"Could not find a Models folder containing {MODEL_FILE}."
    try:
        model = keras.models.load_model(models_dir / MODEL_FILE)
        scaler = joblib.load(models_dir / "scaler.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)


model, scaler, load_error = load_models()
model_loaded = model is not None and scaler is not None


# Feature engineering + prediction
def calculate_features(data):
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


def predict_risk(features, model, scaler):
    if model is None or scaler is None:
        return None

    # DataFrame keeps column names, same as when the scaler was fitted
    X = pd.DataFrame([features], columns=FEATURE_COLS)
    X_scaled = scaler.transform(X)

    probability = float(model.predict(X_scaled, verbose=0).flatten()[0])
    prediction = int(probability >= DECISION_THRESHOLD)

    if probability < LOW_RISK_THRESHOLD:
        risk_category = "Low Risk"
        recommendation = "✅ Approve Loan"
        risk_class = "risk-low"
    elif probability < HIGH_RISK_THRESHOLD:
        risk_category = "Medium Risk"
        recommendation = "⚠️ Review Manually"
        risk_class = "risk-medium"
    else:
        risk_category = "High Risk"
        recommendation = "❌ Reject Loan"
        risk_class = "risk-high"

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_category": risk_category,
        "recommendation": recommendation,
        "risk_class": risk_class,
    }


# Sidebar navigation
with st.sidebar:
    st.markdown("# 🏦")
    st.title("Navigation")

    menu = st.radio(
        "Choose a page:",
        ["📊 Predict Loan Risk", "🤖 AI Copilot"],
        label_visibility="collapsed",
    )

    st.divider()


#  Predict loan risk
if menu == "📊 Predict Loan Risk":
    st.title("📊 Loan Risk Prediction")
    st.markdown("Enter loan application details to get instant default risk assessment")

    if not model_loaded:
        st.error(
            "❌ Model not loaded! Please run `project.py` first to train and "
            f"save models. Details: {load_error}"
        )
        st.stop()

    with st.form("loan_application_form"):
        st.subheader("💼 Loan Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            loan_amount = st.number_input("Loan Amount (₹)", min_value=10000, max_value=10000000, value=500000, step=10000)
            loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=120, value=36, step=6)
            interest_rate = st.number_input("Interest Rate (%)", min_value=5.0, max_value=30.0, value=12.5, step=0.5)

        with col2:
            monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=500000, value=75000, step=5000)
            existing_debt = st.number_input("Existing Debt (₹)", min_value=0, max_value=5000000, value=150000, step=10000)
            previous_defaults = st.number_input("Previous Defaults", min_value=0, max_value=10, value=0, step=1)

        with col3:
            late_payments = st.number_input("Late Payments", min_value=0, max_value=50, value=2, step=1)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=10)
            account_tenure = st.number_input("Account Tenure (years)", min_value=0, max_value=50, value=5, step=1)

        st.subheader("👤 Applicant Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=32, step=1)
            income = st.number_input("Annual Income (₹)", min_value=100000, max_value=2000000, value=900000, step=50000)

        with col2:
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=2, step=1)
            employment_type = st.selectbox("Employment Type", list(EMPLOYMENT_MAP.keys()))

        with col3:
            st.markdown("### 📋 Quick Stats")
            dt_ratio = existing_debt / monthly_income if monthly_income > 0 else 0
            lt_ratio = loan_amount / (monthly_income * 12) if monthly_income > 0 else 0
            st.metric("Debt-to-Income", f"{dt_ratio:.2f}")
            st.metric("Loan-to-Income", f"{lt_ratio:.2f}")

        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Analyzing loan application..."):
            data = {
                "loan_amount": loan_amount,
                "loan_term": loan_term,
                "interest_rate": interest_rate,
                "monthly_income": monthly_income,
                "existing_debt": existing_debt,
                "previous_defaults": previous_defaults,
                "late_payments": late_payments,
                "age": age,
                "income": income,
                "credit_score": credit_score,
                "account_tenure": account_tenure,
                "dependents": dependents,
                "employment_type": employment_type,
            }

            data = calculate_features(data)
            features = [data[col] for col in FEATURE_COLS]
            result = predict_risk(features, model, scaler)

            if result:
                st.markdown("---")

                st.markdown(
                    f'<div class="{result["risk_class"]}">'
                    f'<h2>{result["recommendation"]}</h2>'
                    f'<h3>Risk Category: {result["risk_category"]}</h3>'
                    f'<p>Default Probability: {result["probability"] * 100:.2f}%</p>'
                    "</div>",
                    unsafe_allow_html=True,
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Default Probability", f"{result['probability'] * 100:.2f}%")
                with col2:
                    st.metric("Risk Score", f"{data['risk_score']:.2f}")
                with col3:
                    st.metric("Debt-to-Income", f"{data['debt_to_income']:.2f}")
                with col4:
                    st.metric("Credit Score", data["credit_score"])

                st.markdown("### 📊 Key Risk Factors")

                risk_factors = []
                if data["credit_score"] < 650:
                    risk_factors.append("⚠️ Low Credit Score")
                if data["debt_to_income"] > 3:
                    risk_factors.append("⚠️ High Debt-to-Income Ratio")
                if previous_defaults > 0:
                    risk_factors.append(f"⚠️ {previous_defaults} Previous Default(s)")
                if late_payments > 5:
                    risk_factors.append(f"⚠️ {late_payments} Late Payment(s)")
                if data["loan_to_income"] > 1:
                    risk_factors.append("⚠️ High Loan-to-Income Ratio")

                if risk_factors:
                    for factor in risk_factors:
                        st.warning(factor)
                else:
                    st.success("✅ No major risk factors identified")

                with st.expander("📋 View Application Summary"):
                    st.json(data)

    st.markdown("---")
    st.subheader("📝 Try Example Applications")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👨‍💼 Low Risk Profile", use_container_width=True):
            st.session_state.example = "low_risk"
            st.rerun()

    with col2:
        if st.button("⚠️ Medium Risk Profile", use_container_width=True):
            st.session_state.example = "medium_risk"
            st.rerun()

    with col3:
        if st.button("🚨 High Risk Profile", use_container_width=True):
            st.session_state.example = "high_risk"
            st.rerun()

    if "example" in st.session_state:
        if st.session_state.example == "low_risk":
            st.info("**Low Risk Example:** Credit Score: 780, Income: ₹1,20,000/month, No defaults")
        elif st.session_state.example == "medium_risk":
            st.info("**Medium Risk Example:** Credit Score: 650, Income: ₹60,000/month, 1 default")
        elif st.session_state.example == "high_risk":
            st.info("**High Risk Example:** Credit Score: 520, Income: ₹35,000/month, 3 defaults")


#  AI Copilot
elif menu == "🤖 AI Copilot":
    st.title("🤖 AI Banking Risk Copilot")
    st.markdown(
        "Ask questions about credit risk, fraud detection, "
        "machine learning, and model explainability."
    )

    st.divider()

    render_chatbot()
