import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os

np.random.seed(42)
n_samples = 8760

hours       = np.arange(n_samples)
time_of_day = hours % 24
day_of_week = (hours // 24) % 7
month       = (hours // 720) % 12

base = (
    200
    + 80  * np.sin(2 * np.pi * time_of_day / 24)
    + 30  * np.sin(2 * np.pi * day_of_week / 7)
    + 20  * np.sin(2 * np.pi * month / 12)
    + np.random.normal(0, 15, n_samples)
)

anomaly_idx = np.random.choice(n_samples, size=120, replace=False)
base[anomaly_idx] *= np.random.uniform(1.8, 3.0, size=120)
labels = np.zeros(n_samples)
labels[anomaly_idx] = 1

df = pd.DataFrame({
    "hour":        time_of_day,
    "day_of_week": day_of_week,
    "month":       month,
    "temperature": 20 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 2, n_samples),
    "occupancy":   ((time_of_day >= 8) & (time_of_day <= 18) & (day_of_week < 5)).astype(int),
    "consumption": base,
    "is_anomaly":  labels
})

print(f"Dataset: {df.shape} | Anomaly rate: {df['is_anomaly'].mean():.3f}")

feature_cols = ["hour", "day_of_week", "month", "temperature", "occupancy"]
X = df[feature_cols].values
y = df["consumption"].values
train_size = int(0.8 * n_samples)

mlflow.set_experiment("energy-anomaly-detection")

with mlflow.start_run(run_name="xgboost-forecaster"):
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42)
    xgb.fit(X[:train_size], y[:train_size])
    y_pred = xgb.predict(X[train_size:])
    mae  = mean_absolute_error(y[train_size:], y_pred)
    rmse = np.sqrt(mean_squared_error(y[train_size:], y_pred))
    mlflow.log_metrics({"mae": mae, "rmse": rmse})
    print(f"XGBoost MAE: {mae:.2f}  RMSE: {rmse:.2f}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with mlflow.start_run(run_name="isolation-forest"):
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_scaled[:train_size])
    scores   = iso.decision_function(X_scaled[train_size:])
    preds    = (iso.predict(X_scaled[train_size:]) == -1).astype(int)
    true_bin = labels[train_size:].astype(int)
    from sklearn.metrics import roc_auc_score, classification_report
    auc = roc_auc_score(true_bin, -scores)
    mlflow.log_metrics({"auc_roc": auc})
    print(f"Isolation Forest AUC-ROC: {auc:.4f}")
    print(classification_report(true_bin, preds, target_names=["Normal","Anomaly"]))

os.makedirs("models", exist_ok=True)
joblib.dump(xgb,    "models/xgb_forecaster.joblib")
joblib.dump(iso,    "models/iso_forest.joblib")
joblib.dump(scaler, "models/scaler.joblib")

import json
config = {"mae": float(mae), "rmse": float(rmse), "auc_roc": float(auc), "feature_cols": feature_cols}
with open("models/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("\nModels saved. Config:", json.dumps(config, indent=2))