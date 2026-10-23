
import os
from pathlib import Path

import joblib
import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
from keras import layers
from keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier

SEED = 42
# EarlyStopping will stop earlier if needed
ANN_MAX_EPOCHS = 50          
ANN_BATCH_SIZE = 64
RUN_STABILITY_CHECK = False  
N_ANN_SEEDS = 5
# SHAP for the ANN 
RUN_ANN_SHAP = False
# True = plt.show()
SHOW_PLOTS = False    
SHOW_DATA_PREVIEW = False
STRICT_DETERMINISM = False    


# Same seed everywhere, so every run gives the same result
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "Data").exists():
    BASE_DIR = BASE_DIR.parent

data_dir = BASE_DIR / "Data"
results_dir = BASE_DIR / "Results"
models_dir = BASE_DIR / "Models"

results_dir.mkdir(parents=True, exist_ok=True)
models_dir.mkdir(parents=True, exist_ok=True)

loan = pd.read_csv(data_dir / "loans.csv")
customer = pd.read_csv(data_dir / "customers.csv")
transaction = pd.read_csv(data_dir / "transactions.csv")


for table_name, table in {
    "loan": loan,
    "customer": customer,
    "transaction": transaction,
}.items():
    print(f"\n===== {table_name.upper()} =====")
    print("Shape:", table.shape)
    print(table.head())
    print("Missing values:\n", table.isnull().sum())
transaction["transaction_time"] = pd.to_datetime(transaction["transaction_time"])
transaction["hour"] = transaction["transaction_time"].dt.hour
transaction["is_night"] = (
    (transaction["hour"] >= 23) | (transaction["hour"] <= 5)
).astype(int)

customer_avg = (
    transaction.groupby("customer_id")["amount"]
    .mean()
    .reset_index()
    .rename(columns={"amount": "customer_avg_transaction"})
)
transaction = transaction.merge(customer_avg, on="customer_id", how="left")
transaction["amount_ratio"] = (
    transaction["amount"] / transaction["customer_avg_transaction"]
)
transaction["high_amount"] = (
    transaction["amount"] > transaction["customer_avg_transaction"] * 3
).astype(int)


EMPLOYMENT_MAP = {
    "Salaried": 0,
    "Self-employed": 1,
    "Student": 2,
    "Unemployed": 3,
    "Retired": 4,
}

df = loan.merge(customer, on="customer_id", how="left")

df["debt_to_income"] = df["existing_debt"] / df["monthly_income"]
df["loan_to_income"] = df["loan_amount"] / (df["monthly_income"] * 12)
df["payment_burden"] = df["loan_amount"] / df["loan_term"]
df["risk_score"] = (
    0.35 * (850 - df["credit_score"]) / 550
    + 0.30 * df["debt_to_income"]
    + 0.20 * df["previous_defaults"]
    + 0.15 * df["late_payments"]
)
df["employment_type_encoded"] = df["employment_type"].map(EMPLOYMENT_MAP)

# This order MUST match feature_cols
feature_cols = [
    "loan_amount", "loan_term", "interest_rate", "monthly_income",
    "existing_debt", "previous_defaults", "late_payments",
    "age", "income", "credit_score", "account_tenure", "dependents",
    "debt_to_income", "loan_to_income", "payment_burden", "risk_score",
    "employment_type_encoded",
]

# Clean
df = df.replace([np.inf, -np.inf], np.nan)
rows_before = len(df)
df = df.dropna(subset=feature_cols + ["default"])
print(f"\nRows dropped because of missing/invalid values: {rows_before - len(df)}")

X = df[feature_cols]
y = df["default"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Scaler learns only from train data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class imbalance
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()
scale_pos_weight = negative / positive
print("\nScale Pos Weight:", round(scale_pos_weight, 3))

weights = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=y_train
)
ann_class_weight = {0: weights[0], 1: weights[1]}



def evaluate(y_true, proba, threshold=0.5):
    """Return all metrics for one model."""
    pred = (proba >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, proba),
        "Avg Precision": average_precision_score(y_true, proba),
    }


def print_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


def finish_plot(filename):
    """Save the current figure, optionally show it, then close it."""
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def build_ann(n_features):
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_ann(seed, verbose=0):
    """Train one ANN with a given seed. Returns (model, history)."""
    keras.utils.set_random_seed(seed)
    model = build_ann(X_train_scaled.shape[1])
    early_stop = EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=ANN_MAX_EPOCHS,
        batch_size=ANN_BATCH_SIZE,
        callbacks=[early_stop],
        class_weight=ann_class_weight,
        verbose=verbose,
    )
    return model, history


# Models: Logistic  Random Forest, XGBoost
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=SEED, class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", random_state=SEED,
    ),
}

results = []
print("\nMODEL COMPARISON - CREDIT RISK PREDICTION")

for name, model in models.items():
    print(f"\nTraining: {name}")

    # Only Logistic Regression needs scaled data 
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, proba)
    results.append({"Model": name, **metrics})
    print_metrics(metrics)

# ANN
print("\nTraining: ANN (Neural Network)")
ann_model, history = train_ann(SEED, verbose=1)
epochs_trained = len(history.history["loss"])
print(f"ANN stopped after {epochs_trained} epochs")

ann_proba = ann_model.predict(X_test_scaled, verbose=0).flatten()
ann_metrics = evaluate(y_test, ann_proba)
results.append({"Model": "ANN (Neural Network)", **ann_metrics})
print("\nANN Results:")
print_metrics(ann_metrics)

results_df = pd.DataFrame(results)
print("\nFINAL COMPARISON TABLE (single run, seed fixed)")
print(results_df.sort_values("ROC-AUC", ascending=False).to_string(index=False))

# stability check
ann_auc_mean = ann_auc_std = None

if RUN_STABILITY_CHECK:
    print("\nSTABILITY CHECK")

    ann_aucs = [ann_metrics["ROC-AUC"]]
    for i in range(1, N_ANN_SEEDS):
        extra_model, _ = train_ann(SEED + i, verbose=0)
        extra_proba = extra_model.predict(X_test_scaled, verbose=0).flatten()
        ann_aucs.append(roc_auc_score(y_test, extra_proba))

    ann_auc_mean = float(np.mean(ann_aucs))
    ann_auc_std = float(np.std(ann_aucs))
    print(f"ANN test ROC-AUC over {N_ANN_SEEDS} seeds: "
          f"{ann_auc_mean:.4f} +/- {ann_auc_std:.4f}")

    #  5-fold cross-validation on the train set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, model in models.items():
        estimator = (
            make_pipeline(StandardScaler(), model)
            if name == "Logistic Regression"
             else model
        )
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv, scoring="roc_auc"
        )
        print(f"{name} CV ROC-AUC: {scores.mean():.4f} +/- {scores.std():.4f}")

    best_other = results_df[results_df["Model"] != "ANN (Neural Network)"][
        "ROC-AUC"
    ].max()
    if ann_auc_mean - best_other > ann_auc_std:
        print("\nVerdict: ANN is ahead by more than its own seed-to-seed noise.")
    else:
        print("\nVerdict: the gap is within noise. Models perform about the "
              "same, so do not claim one 'best' model.")

# SHAP 1: Logistic Regression 
print("\nSHAP MODEL EXPLAINABILITY")

X_sample = X_test.sample(500, random_state=SEED)
X_sample_scaled = scaler.transform(X_sample)

lr_explainer = shap.LinearExplainer(
    models["Logistic Regression"], X_train_scaled, feature_names=feature_cols
)
shap_values = lr_explainer.shap_values(X_sample_scaled)
print(f"SHAP values shape: {shap_values.shape}")

shap.summary_plot(
    shap_values, X_sample_scaled, feature_names=feature_cols,
    plot_type="bar", show=False, plot_size=(10, 8),
)
plt.title("Feature Importance (SHAP - Logistic Regression)",
          fontsize=14, fontweight="bold")
finish_plot("shap_feature_importance.png")

# Beeswarm: 
shap.summary_plot(
    shap_values, X_sample_scaled, feature_names=feature_cols,
    plot_type="dot", show=False, plot_size=(10, 8),
)
plt.title("SHAP Values - Feature Impact on Predictions",
          fontsize=14, fontweight="bold")
finish_plot("shap_summary_beeswarm.png")

# Single prediction
idx = 0
shap_df = (
    pd.DataFrame({"Feature": feature_cols, "SHAP Value": shap_values[idx]})
    .sort_values("SHAP Value", key=np.abs, ascending=False)
)
print(f"\nPrediction {idx} SHAP values (top 5 features):")
print(shap_df.head(5).to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    data=shap_df.head(10), x="SHAP Value", y="Feature",
    hue="Feature", palette="viridis", legend=False,
)
plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
plt.title(f"Top 10 Feature Contributions - Prediction {idx}",
          fontsize=14, fontweight="bold")
finish_plot("shap_single_prediction.png")

if RUN_ANN_SHAP:
    print("\nCalculating SHAP for ANN (KernelExplainer, small sample)...")
    background = shap.kmeans(X_train_scaled, 20)
    ann_explainer = shap.KernelExplainer(
        lambda data: ann_model.predict(data, verbose=0).flatten(), background
    )
    X_ann_sample = X_sample_scaled[:100]
    ann_shap = np.array(ann_explainer.shap_values(X_ann_sample, nsamples=200))
    if ann_shap.ndim == 3:
        ann_shap = ann_shap[..., 0]

    shap.summary_plot(
        ann_shap, X_ann_sample, feature_names=feature_cols,
        plot_type="bar", show=False, plot_size=(10, 8),
    )
    plt.title("Feature Importance (SHAP - ANN)", fontsize=14, fontweight="bold")
    finish_plot("shap_ann_feature_importance.png")

print("\nSHAP analysis complete. Check the Results folder for plots.")

# mlflow
mlflow.set_experiment("Credit_Risk_Model_Comparison")

with mlflow.start_run():
    best_row = results_df.loc[results_df["ROC-AUC"].idxmax()]

    mlflow.log_params({
        "best_model_single_run": best_row["Model"],
        "seed": SEED,
        "test_size": 0.2,
        "xgb_n_estimators": 300,
        "xgb_max_depth": 6,
        "xgb_learning_rate": 0.05,
        "rf_n_estimators": 300,
        "rf_max_depth": 12,
        "ann_architecture": "64-32-16-1 with dropout",
        "ann_max_epochs": ANN_MAX_EPOCHS,
        "ann_epochs_trained": epochs_trained,
        "ann_batch_size": ANN_BATCH_SIZE,
    })

    mlflow.log_metric("best_roc_auc", best_row["ROC-AUC"])
    mlflow.log_metric("best_f1", best_row["F1 Score"])
    mlflow.log_metric("best_precision", best_row["Precision"])
    mlflow.log_metric("best_recall", best_row["Recall"])

    for _, row in results_df.iterrows():
        tag = (row["Model"].replace(" ", "_")
               .replace("(", "").replace(")", ""))
        mlflow.log_metric(f"{tag}_roc_auc", row["ROC-AUC"])
        mlflow.log_metric(f"{tag}_f1", row["F1 Score"])
        mlflow.log_metric(f"{tag}_precision", row["Precision"])
        mlflow.log_metric(f"{tag}_recall", row["Recall"])

    if ann_auc_mean is not None:
        mlflow.log_metric("ann_roc_auc_mean_over_seeds", ann_auc_mean)
        mlflow.log_metric("ann_roc_auc_std_over_seeds", ann_auc_std)

    for png in results_dir.glob("*.png"):
        mlflow.log_artifact(str(png))

    print("\nMLflow logging complete!")
    print(f"Best model (single run): {best_row['Model']} "
          f"(ROC-AUC: {best_row['ROC-AUC']:.4f})")

# Save models
ann_model.save(models_dir / "credit_risk_ann.keras")
joblib.dump(models["XGBoost"], models_dir / "credit_risk_xgb.pkl")
joblib.dump(models["Random Forest"], models_dir / "credit_risk_rf.pkl")
joblib.dump(models["Logistic Regression"], models_dir / "credit_risk_lr.pkl")
joblib.dump(scaler, models_dir / "scaler.pkl")

print("\nAll models saved successfully!")
print(f"Models saved to: {models_dir}")