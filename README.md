# 🏦 AI Banking Fraud & Credit Risk Platform

> **An end-to-end Machine Learning platform for intelligent fraud detection and financial risk analysis, combining ML, Explainable AI, experiment tracking, API serving, and containerized deployment.**

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP-purple)
![MLflow](https://img.shields.io/badge/MLOps-MLflow-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED?logo=docker)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)
![Status](https://img.shields.io/badge/Project-Portfolio%20Project-success)

---

## 📌 Project Overview

The **AI Banking Fraud & Credit Risk Platform** is an end-to-end AI/ML project designed to demonstrate how machine learning can be applied to financial security and risk assessment.

The platform focuses on two important banking use cases:

- 🔴 **Fraud Detection** — identifying potentially fraudulent financial transactions.
- 🟡 **Credit/Risk Analysis** — estimating financial risk based on customer and transaction-related features.

Rather than stopping at model training, the project follows an engineering-oriented workflow:

```text
Data
  ↓
Data Cleaning & Validation
  ↓
Exploratory Data Analysis
  ↓
Feature Engineering
  ↓
Model Training
  ↓
Model Evaluation
  ↓
Explainable AI (SHAP)
  ↓
MLflow Experiment Tracking
  ↓
FastAPI Model Serving
  ↓
Docker
  ↓
Streamlit Dashboard
```

The goal is to demonstrate the complete journey from **raw financial data to an explainable machine-learning prediction system**.

---

# 🎯 Objectives

The main objectives of this project are:

- Detect potentially fraudulent transactions.
- Analyze financial/customer risk.
- Perform meaningful feature engineering.
- Handle class imbalance in fraud detection.
- Compare machine-learning models using appropriate metrics.
- Use probability-based risk scoring rather than relying only on accuracy.
- Provide model explanations using **SHAP**.
- Track experiments and model artifacts using **MLflow**.
- Serve trained models through an API layer using **FastAPI**.
- Containerize the application using **Docker**.
- Provide an interactive interface using **Streamlit**.
- Build a foundation for a real-time financial intelligence system.

---

# 🚨 Why Fraud Detection Is Challenging

Fraud detection is fundamentally different from many standard classification problems.

In a typical banking dataset:

```text
Legitimate transactions  >>>  Fraudulent transactions
```

This creates **class imbalance**.

For example, a model could achieve very high accuracy simply by predicting most transactions as legitimate.

Therefore, this project focuses on metrics that are more meaningful for fraud detection:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC / Average Precision
- Confusion Matrix

### Why Recall Matters

For fraud detection:

> Missing a fraudulent transaction can be much more costly than investigating a legitimate transaction.

Therefore, the model should not be evaluated using accuracy alone.

---

# 🧠 Machine Learning Pipeline

## 1. Data Ingestion

Financial transaction/customer data is loaded and prepared for analysis.

```text
Raw Data
   ↓
Validation
   ↓
Cleaning
   ↓
Transformation
```

---

## 2. Data Preprocessing

The preprocessing pipeline handles tasks such as:

- Missing-value handling
- Categorical encoding
- Numerical scaling
- Feature transformation
- Data validation
- Train/test splitting

The objective is to ensure that the same preprocessing logic can be applied consistently during inference.

---

# ⚙️ Feature Engineering

Feature engineering is one of the most important parts of this project.

Instead of relying only on raw transaction fields, additional behavioral and risk-related features can be derived.

Examples include:

### Transaction Amount Ratio

```text
amount_ratio =
transaction_amount / average_transaction_amount
```

This helps identify transactions that are unusually large compared with historical behavior.

### Unusual Transaction Hour

Transactions occurring during unusual hours can contribute to a higher risk score.

### High-Amount Indicator

A binary feature can identify unusually large transactions.

### Risk Score

Multiple behavioral signals can be combined into a derived risk-related feature.

Feature engineering allows the model to learn **patterns of suspicious behavior**, rather than simply memorizing individual transaction values.

---

# 🤖 Machine Learning Models

The project is designed around classification models suitable for tabular financial data.

Potential model families include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting / XGBoost
- Other suitable classification models

The models are compared using fraud-focused evaluation metrics rather than relying only on accuracy.

---

# 📊 Model Evaluation

The following metrics are used to evaluate model performance:

| Metric | Purpose |
|---|---|
| Precision | How many predicted fraud cases were actually fraud |
| Recall | How many actual fraud cases were detected |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Overall ranking capability |
| PR-AUC | Useful for imbalanced fraud datasets |
| Confusion Matrix | Detailed classification analysis |

### Example Evaluation Concept

```text
                 Actual
              Fraud   Legit
Pred Fraud      TP      FP
Pred Legit      FN      TN
```

The objective is to reduce dangerous **False Negatives (FN)** while maintaining an acceptable false-positive rate.

---

# 🔍 Explainable AI with SHAP

A major component of the platform is **Explainable AI (XAI)** using SHAP.

Traditional ML prediction:

```text
Transaction → Fraud
```

Explainable prediction:

```text
Transaction
     ↓
Fraud Probability: 0.91
     ↓
SHAP Explanation
     ↓
┌──────────────────────────────┐
│ High transaction amount      │ → Higher risk
│ Unusual transaction time     │ → Higher risk
│ High transaction ratio       │ → Higher risk
│ Normal historical behavior   │ → Lower risk
└──────────────────────────────┘
```

SHAP helps answer an important banking question:

> **"Why did the model classify this transaction as risky?"**

This makes the system more interpretable and useful for financial-risk analysis.

---

# 🧪 MLflow Experiment Tracking

The project integrates **MLflow** to support the machine-learning lifecycle.

MLflow can be used to track:

- Experiments
- Parameters
- Metrics
- Model artifacts
- Model versions
- Training runs

Conceptually:

```text
Experiment
    ↓
Training Run
    ├── Parameters
    ├── Metrics
    ├── Artifacts
    └── Model
```

This makes it easier to compare different experiments and reproduce model-development results.

The repository also contains an `mlruns` directory for MLflow artifacts.

---

# 🚀 FastAPI Model Serving

The project includes a FastAPI layer for exposing machine-learning functionality through an API.

Conceptual inference flow:

```text
Client / Banking System
          ↓
       FastAPI
          ↓
 Feature Engineering
          ↓
    Trained Model
          ↓
   Fraud Probability
          ↓
    Risk Decision
          ↓
     API Response
```

A production-oriented API can accept transaction information and return:

```json
{
  "prediction": "fraud",
  "fraud_probability": 0.91,
  "risk_level": "high"
}
```

> **Note:** Keep the exact endpoint names and request schema synchronized with the implementation in the repository.

---

# 🖥️ Streamlit Dashboard

The Streamlit application provides an interactive interface for demonstrating model predictions.

Possible workflow:

```text
Enter Transaction Details
          ↓
Submit Prediction
          ↓
ML Model
          ↓
Fraud Probability
          ↓
Risk Classification
          ↓
SHAP Explanation
```

The repository contains a dedicated `Streamlit` directory and a separate Streamlit Docker configuration.

---

# 🐳 Docker Deployment

The project is containerized using Docker.

Repository components include:

```text
Dockerfile
Dockerfile.streamlit
docker-compose.yml
.dockerignore
build_and_run.bat
```

This allows the application environment to be packaged consistently across machines.

### Container Architecture

```text
                 Docker Environment
                        │
            ┌───────────┴───────────┐
            ↓                       ↓
       ML/API Service          Streamlit UI
            │                       │
            └───────────┬───────────┘
                        ↓
                  ML Model
```

---

# 📁 Project Structure

```text
AI-Banking-Fraud-Credit-Risk-Platform/
│
├── Data/
│   └── Dataset files
│
├── Models/
│   └── Trained model artifacts
│
├── Results/
│   ├── Evaluation results
│   └── Analysis outputs
│
├── Streamlit/
│   └── Streamlit application
│
├── src/
│   ├── Data processing
│   ├── Feature engineering
│   ├── Model training
│   └── Prediction logic
│
├── mlruns/
│   └── MLflow experiment artifacts
│
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── .dockerignore
├── build_and_run.bat
├── requirements.txt
├── requirements_streamlit.txt
└── README.md
```

The current GitHub repository contains these major project directories and deployment files.

---

# 🛠️ Technology Stack

## Programming

- Python

## Data Science

- Pandas
- NumPy
- Matplotlib
- Seaborn

## Machine Learning

- Scikit-learn
- XGBoost

## Explainable AI

- SHAP

## MLOps

- MLflow

## API

- FastAPI

## Deployment

- Docker
- Docker Compose

## UI

- Streamlit

---

# ⚙️ Installation

## 1. Clone the repository

```bash
git clone https://github.com/sonianurag951-oss/AI-Banking-Fraud-Credit-Risk-Platform.git
```

```bash
cd AI-Banking-Fraud-Credit-Risk-Platform
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For the Streamlit application:

```bash
pip install -r requirements_streamlit.txt
```

---

# ▶️ Running the Project

## Train the model

Run the appropriate training script from the `src` directory:

```bash
python src/train.py
```

> Adjust the filename if the training entry point in your repository uses a different name.

---

## Run Streamlit

From the Streamlit directory:

```bash
streamlit run app.py
```

---

## Run FastAPI

If your API entry point is `main.py`:

```bash
uvicorn main:app --reload
```

The API documentation can then be accessed through FastAPI's interactive documentation.

> Update this command to match the actual API entry-point file in the repository.

---

# 🐳 Running with Docker

Build the application:

```bash
docker build -t ai-banking-fraud-platform .
```

Run the container:

```bash
docker run -p 8000:8000 ai-banking-fraud-platform
```

For a multi-service setup:

```bash
docker compose up --build
```

Stop the services:

```bash
docker compose down
```

---

# 🔄 End-to-End Prediction Architecture

The intended architecture of the platform is:

```text
                  ┌──────────────────────┐
                  │  Banking Transaction │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │       FastAPI        │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │ Feature Engineering │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │    ML Prediction     │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   Fraud Probability  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │    SHAP Explanation  │
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   Risk Classification│
                  └──────────┬───────────┘
                             ↓
                  ┌──────────────────────┐
                  │   Streamlit / Client │
                  └──────────────────────┘
```

---

# 📈 Future Improvements

The platform can be extended toward a more production-oriented financial intelligence system.

### Planned Improvements

- [ ] PostgreSQL transaction storage
- [ ] Real-time transaction ingestion
- [ ] Model monitoring
- [ ] Data drift detection
- [ ] Automated retraining
- [ ] CI/CD pipeline
- [ ] Authentication and authorization
- [ ] Transaction history dashboard
- [ ] Alert/notification system
- [ ] Model version management
- [ ] Cloud deployment
- [ ] Kafka/event-streaming integration for high-volume transaction processing

---

# 🔐 Security & Responsible Use

This project is intended for **educational, portfolio, and demonstration purposes**.

A production banking system would require additional controls, including:

- Secure authentication
- Authorization
- Encryption
- Sensitive-data protection
- Audit logging
- Model governance
- Bias/fairness evaluation
- Regulatory compliance
- Human review for high-risk decisions

The model should not be treated as the sole decision-maker for real financial transactions.

---

# 💡 Key Learning Outcomes

Through this project, I worked with concepts across the complete machine-learning lifecycle:

### Machine Learning

- Classification
- Feature engineering
- Imbalanced datasets
- Model comparison
- Probability-based predictions
- Evaluation metrics

### Explainable AI

- SHAP
- Feature importance
- Individual prediction explanations

### MLOps

- MLflow
- Experiment tracking
- Model artifacts
- Reproducibility

### Deployment

- FastAPI
- Streamlit
- Docker
- Docker Compose

### Software Engineering

- Modular project structure
- Separation of training and inference
- Dependency management
- Containerized execution

---

# 🎯 Why This Project Matters

This project goes beyond a simple:

```text
Dataset → Model → Accuracy
```

It demonstrates an end-to-end AI/ML workflow:

```text
Data
 ↓
Engineering
 ↓
Machine Learning
 ↓
Evaluation
 ↓
Explainability
 ↓
Experiment Tracking
 ↓
API
 ↓
Containerization
 ↓
Interactive Application
```

This makes the project suitable for demonstrating practical skills for **AI/ML Engineer, Machine Learning Engineer, and Data Science fresher roles**.

---

# 👨‍💻 Resume Highlights

You can describe the project on your resume as:

> **AI Banking Fraud & Credit Risk Platform** — Developed an end-to-end machine learning platform for financial fraud and risk analysis using Python, Scikit-learn/XGBoost, SHAP, MLflow, FastAPI, Docker, and Streamlit. Implemented feature engineering and fraud-focused evaluation using precision, recall, F1-score, ROC-AUC, and explainable AI techniques.

### Stronger interview description

> Built an end-to-end banking risk platform that takes transaction-level data through preprocessing and feature engineering, generates ML-based fraud predictions, explains predictions using SHAP, tracks experiments with MLflow, exposes inference through FastAPI, and provides an interactive Streamlit interface with Docker-based deployment.

---

# 📌 Project Status

**Current Stage:** Active Development 🚧

The project is being developed incrementally toward a complete real-world financial intelligence pipeline.

---

# ⭐ Author

**Anurag Soni**

B.Tech — Artificial Intelligence / Machine Learning

Interested in:

- Machine Learning
- Deep Learning
- MLOps
- Explainable AI
- AI Engineering
- Data Science

---

# 📄 License

This project is intended for educational and portfolio purposes.

Add an appropriate open-source license if you plan to distribute the project for reuse.