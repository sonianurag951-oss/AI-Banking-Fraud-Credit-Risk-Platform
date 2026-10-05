import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import shap
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from keras.callbacks import EarlyStopping
import keras
from keras import layers
from xgboost import XGBClassifier
import mlflow

import mlflow.tensorflow
import mlflow.sklearn 
loan = pd.read_csv(r"D:\deep learning\AI Banking Fraud & Credit-Risk Platform\Data\loans.csv")
customer = pd.read_csv(r"D:\deep learning\AI Banking Fraud & Credit-Risk Platform\Data\customers.csv")
transaction = pd.read_csv(r"D:\deep learning\AI Banking Fraud & Credit-Risk Platform\Data\transactions.csv")


results_dir = r"D:\deep learning\AI Banking Fraud & Credit-Risk Platform\Results"
models_dir = r"D:\deep learning\AI Banking Fraud & Credit-Risk Platform\Models"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)


# data cleaning 

print(loan.info())
print(loan.head(10))
print(loan.describe())
print(loan.isnull().sum())
print(loan.shape)


print(customer.info())
print(customer.head(10))
print(customer.describe())
print(customer.isnull().sum())
print(customer.shape)

print(transaction.info())
print(transaction.head(10))
print(transaction.describe())
print(transaction.isnull().sum())
print(transaction.shape)
# Convert transaction_time to datetime
transaction['transaction_time'] = pd.to_datetime(transaction['transaction_time'])

# Feature engineering 
transaction["hour"] = transaction["transaction_time"].dt.hour
transaction["is_night"] = ((transaction["hour"] >= 23) | (transaction["hour"] <= 5)).astype(int)

# Calculate customer average transaction FIRST
customer_avg = transaction.groupby('customer_id')['amount'].mean().reset_index()
customer_avg.columns = ['customer_id', 'customer_avg_transaction']

# Merge back to transaction
transaction = transaction.merge(customer_avg, on='customer_id', how='left')

transaction["amount_ratio"] = transaction["amount"] / transaction["customer_avg_transaction"]
transaction["high_amount"] = (transaction["amount"] > transaction["customer_avg_transaction"] * 3).astype(int)

# Feature engineering 
df = loan.merge(customer, on='customer_id', how='left')

df["debt_to_income"] = df["existing_debt"] / df["monthly_income"]
df["loan_to_income"] = df["loan_amount"] / (df["monthly_income"] * 12)
df["payment_burden"] = df["loan_amount"] / df["loan_term"]

df["risk_score"] = (
    0.35 * (850 - df["credit_score"]) / 550
    + 0.30 * df["debt_to_income"]
    + 0.20 * df["previous_defaults"]
    + 0.15 * df["late_payments"]
)
print(transaction.head())
print(df.head())

# Features
feature_cols = [
    'loan_amount', 'loan_term', 'interest_rate', 'monthly_income',
    'existing_debt', 'previous_defaults', 'late_payments',
    'age', 'income', 'credit_score', 'account_tenure', 'dependents',
    'debt_to_income', 'loan_to_income', 'payment_burden', 'risk_score'
]

# Encode categorical
df['employment_type_encoded'] = df['employment_type'].map({
    'Salaried': 0, 'Self-employed': 1, 'Student': 2, 
    'Unemployed': 3, 'Retired': 4
})
feature_cols.append('employment_type_encoded')

X=df[feature_cols]
y=df['default']
# train test split


X_train, X_test, y_train, y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standard scale for ANN AND LOGISTIC 
scaled=StandardScaler()
X_train_scaled=scaled.fit_transform(X_train)
X_test_scaled=scaled.transform(X_test)

# class imbalance

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nScale Pos Weight:", scale_pos_weight)

# DEFINE ALL 4 MODELS
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5, 
        min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', random_state=42
    )
}

# TRAIN & EVALUATE ALL MODELS 
results = []

print("\n" + "="*70)
print("MODEL COMPARISON - CREDIT RISK PREDICTION")
print("="*70)

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    
    # Fit model
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)
    avg_prec = average_precision_score(y_test, pred_proba)
    
    results.append({
        'Model': name,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC-AUC': roc_auc,
        'Avg Precision': avg_prec
    })
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Avg Precision: {avg_prec:.4f}")

#  ANN MODEL
print("\n" + "="*70)
print("Training: ANN (Neural Network)")
print("="*70)

EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# Build ANN
ann_model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Compile with class weight for imbalance
ann_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')]
)

# Train
history = ann_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=64,
    verbose=1
)

# Evaluate ANN
ann_pred = (ann_model.predict(X_test_scaled) > 0.5).astype(int).flatten()
ann_pred_proba = ann_model.predict(X_test_scaled).flatten()

ann_precision = precision_score(y_test, ann_pred)
ann_recall = recall_score(y_test, ann_pred)
ann_f1 = f1_score(y_test, ann_pred)
ann_roc_auc = roc_auc_score(y_test, ann_pred_proba)
ann_avg_prec = average_precision_score(y_test, ann_pred_proba)

results.append({
    'Model': 'ANN (Neural Network)',
    'Precision': ann_precision,
    'Recall': ann_recall,
    'F1 Score': ann_f1,
    'ROC-AUC': ann_roc_auc,
    'Avg Precision': ann_avg_prec
})

print(f"\nANN Results:")
print(f"Precision: {ann_precision:.4f}")
print(f"Recall: {ann_recall:.4f}")
print(f"F1 Score: {ann_f1:.4f}")
print(f"ROC-AUC: {ann_roc_auc:.4f}")
print(f"Avg Precision: {ann_avg_prec:.4f}")

# shap value
# RESULTS DATAFRAME
results_df = pd.DataFrame(results)
print("\n" + "="*70)
print("FINAL COMPARISON TABLE")
print("="*70)
print(results_df.to_string(index=False))

print("\n" + "="*70)
print("SHAP MODEL EXPLAINABILITY")
print("="*70)

# Use Logistic Regression (best model)
model_to_explain = models['Logistic Regression']

# For Logistic Regression, use LinearExplainer
print("\nCreating SHAP explainer for Logistic Regression...")
explainer = shap.LinearExplainer(model_to_explain, X_train, feature_names=feature_cols)

# Calculate SHAP values
print("Calculating SHAP values...")
X_sample = X_test.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Check SHAP values shape
print(f"SHAP values shape: {shap_values.shape}")
print(f"X_sample shape: {X_sample.shape}")

# 1. SHAP Summary Plot (Bar)
print("\nGenerating SHAP summary plot (bar)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_cols)
plt.title('Feature Importance (SHAP - Logistic Regression)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "shap_feature_importance.png"), dpi=300)
plt.show()

# 2. SHAP Summary Plot (Beeswarm) - FIXED
print("Generating SHAP summary plot (beeswarm)...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_values, 
    X_sample, 
    feature_names=feature_cols,
    plot_type="dot",  # Use "dot" instead of default beeswarm for LinearExplainer
    color='viridis'
)
plt.title('SHAP Values - Feature Impact on Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "shap_summary_beewarm.png"), dpi=300)
plt.show()

# 3. Single Prediction Explanation
print("Generating single prediction explanation...")
idx = 0
single_shap = shap_values[idx]
single_data = X_sample.iloc[idx]

# Show as text first
print(f"\nPrediction {idx} SHAP values (top 5 features):")
shap_df = pd.DataFrame({
    'Feature': feature_cols,
    'SHAP Value': single_shap
}).sort_values('SHAP Value', key=abs, ascending=False)
print(shap_df.head(5).to_string(index=False))

# Create bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=shap_df.head(10), x='SHAP Value', y='Feature', palette='viridis')
plt.title(f'Top 10 Feature Contributions - Prediction {idx}', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=1)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, "shap_single_prediction.png"), dpi=300)
plt.show()

print("\n SHAP analysis complete! Check Results folder for plots.")

# MLFLOW LOGGING 
mlflow.set_experiment("Credit_Risk_Model_Comparison")

with mlflow.start_run():
    # Log best model
    best_model = results_df.loc[results_df['ROC-AUC'].idxmax()]
    
    # Log parameters
    mlflow.log_param("best_model", best_model['Model'])
    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 6)
    mlflow.log_param("learning_rate", 0.05)
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("best_roc_auc", best_model['ROC-AUC'])
    mlflow.log_metric("best_f1", best_model['F1 Score'])
    mlflow.log_metric("best_precision", best_model['Precision'])
    mlflow.log_metric("best_recall", best_model['Recall'])
    
    # Log all model metrics
    for idx, row in results_df.iterrows():
        model_name = row['Model'].replace(' ', '_').replace('(', '').replace(')', '')
        mlflow.log_metric(f"{model_name}_roc_auc", row['ROC-AUC'])
        mlflow.log_metric(f"{model_name}_f1", row['F1 Score'])
        mlflow.log_metric(f"{model_name}_precision", row['Precision'])
        mlflow.log_metric(f"{model_name}_recall", row['Recall'])
    
    
    # Save ANN model info
    mlflow.log_param("ANN_architecture", "64-32-16-1 with dropout")
    mlflow.log_param("ANN_epochs", 20)
    mlflow.log_param("ANN_batch_size", 64)
    
    print("\n MLflow logging complete!")
    print(f" Best Model: {best_model['Model']} (ROC-AUC: {best_model['ROC-AUC']:.4f})")


import joblib

joblib.dump(models['XGBoost'], os.path.join(models_dir, "credit_risk_xgb.pkl"))
joblib.dump(models['Random Forest'], os.path.join(models_dir, "credit_risk_rf.pkl"))
joblib.dump(models['Logistic Regression'], os.path.join(models_dir, "credit_risk_lr.pkl"))
joblib.dump(scaled, os.path.join(models_dir, "scaler.pkl"))

print("\n✅ All models saved successfully!")
print(f"📁 Models saved to: {models_dir}")