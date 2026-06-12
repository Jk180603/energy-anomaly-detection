from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib, json, numpy as np, time, os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Energy Anomaly Detection API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Prometheus metrics ───────────────────────────────────────────────────────
REQUEST_COUNT    = Counter("predictions_total", "Total predictions", ["result"])
REQUEST_LATENCY  = Histogram("prediction_latency_seconds", "Prediction latency")
ANOMALY_RATE     = Gauge("anomaly_rate", "Current anomaly rate")
CONSUMPTION_GAUGE = Gauge("predicted_consumption_kwh", "Predicted energy consumption")

# ── Load models ──────────────────────────────────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "./models")
xgb    = joblib.load(f"{MODEL_PATH}/xgb_forecaster.joblib")
iso    = joblib.load(f"{MODEL_PATH}/iso_forest.joblib")
scaler = joblib.load(f"{MODEL_PATH}/scaler.joblib")
with open(f"{MODEL_PATH}/config.json") as f:
    config = json.load(f)

anomaly_count = 0
total_count   = 0

class PredictionRequest(BaseModel):
    hour:        float
    day_of_week: float
    month:       float
    temperature: float
    occupancy:   int

class PredictionResponse(BaseModel):
    predicted_consumption: float
    is_anomaly:            bool
    anomaly_score:         float
    alert:                 str

@app.get("/")
def root():
    return {"message": "Energy Anomaly Detection API running", "model_metrics": config}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    global anomaly_count, total_count
    start = time.time()

    features = np.array([[req.hour, req.day_of_week, req.month,
                          req.temperature, req.occupancy]])

    predicted_consumption = float(xgb.predict(features)[0])
    features_scaled       = scaler.transform(features)
    anomaly_score         = float(-iso.decision_function(features_scaled)[0])
    is_anomaly            = bool(iso.predict(features_scaled)[0] == -1)

    total_count   += 1
    if is_anomaly:
        anomaly_count += 1

    ANOMALY_RATE.set(anomaly_count / total_count)
    CONSUMPTION_GAUGE.set(predicted_consumption)
    REQUEST_COUNT.labels(result="anomaly" if is_anomaly else "normal").inc()
    REQUEST_LATENCY.observe(time.time() - start)

    alert = ""
    if is_anomaly:
        alert = f"ANOMALY DETECTED: Score {anomaly_score:.3f} — consumption {predicted_consumption:.1f} kWh"

    return PredictionResponse(
        predicted_consumption=predicted_consumption,
        is_anomaly=is_anomaly,
        anomaly_score=anomaly_score,
        alert=alert
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": True}

@app.get("/stats")
def stats():
    return {
        "total_predictions": total_count,
        "anomaly_count":     anomaly_count,
        "anomaly_rate":      anomaly_count / total_count if total_count > 0 else 0
    }