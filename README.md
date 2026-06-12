# Energy Anomaly Detection for Smart Buildings

A production-style MLOps pipeline that forecasts building energy consumption and detects anomalies in real time, with full monitoring and Kubernetes deployment.

## Overview

This system combines two machine learning models working together:

- **XGBoost Forecaster** predicts expected energy consumption based on time of day, day of week, month, temperature, and occupancy.
- **Isolation Forest** compares actual consumption against the prediction and flags anomalies when behaviour deviates significantly from the learned normal pattern.

The result is anomaly detection with context: not just "this is unusual" but "this is unusual relative to what we expected for this time and these conditions."

## Architecture

```
Request → FastAPI → XGBoost (expected consumption)
                  → Isolation Forest (anomaly score)
                  → Prometheus metrics
                  → Grafana dashboard
                  ↓
            Kubernetes (2 replicas, HPA autoscaling 2-10 pods)
```

## Screenshots

### Grafana Monitoring Dashboard
![Grafana Dashboard](screenshots/grafana-dashboard.png)

Live dashboard tracking total predictions, anomaly rate, prediction latency, and predicted energy consumption.

### FastAPI Swagger UI (running through Kubernetes)
![FastAPI Docs](screenshots/fastapi-docs.png)

### Kubernetes Pods Running
![Kubernetes Pods](screenshots/k8s-pods.png)

2 replicas running with health checks passing, served through `kubectl port-forward`.

## Tech Stack

| Layer | Technology |
|---|---|
| ML Models | XGBoost (forecasting), Isolation Forest (anomaly detection), Scikit-learn |
| Experiment Tracking | MLflow |
| API | FastAPI |
| Monitoring | Prometheus, Grafana |
| Containerisation | Docker, Docker Compose |
| Orchestration | Kubernetes (k3d), HorizontalPodAutoscaler |

## Getting Started

### Train the models

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train.py
```

This generates a synthetic year of hourly building energy data, trains both models, and saves them to `models/`.

### Run locally with Docker Compose

```bash
docker-compose up --build
```

- API: http://localhost:8000/docs
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090

### Deploy to Kubernetes (k3d)

```bash
docker build -t energy-anomaly-api:latest .
k3d cluster create energy-cluster
k3d image import energy-anomaly-api:latest -c energy-cluster
kubectl apply -f k8s/deployment.yaml
kubectl port-forward svc/energy-anomaly-service 8080:80
```

Open http://localhost:8080/docs

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Returns predicted consumption, anomaly flag, and anomaly score |
| `/metrics` | GET | Prometheus metrics endpoint |
| `/health` | GET | Health check used by Kubernetes liveness/readiness probes |
| `/stats` | GET | Running totals of predictions and anomaly rate |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"hour":14,"day_of_week":1,"month":6,"temperature":28,"occupancy":1}'
```

### Example response

```json
{
  "predicted_consumption": 187.4,
  "is_anomaly": false,
  "anomaly_score": 0.04,
  "alert": ""
}
```

## Monitoring

Four Grafana panels backed by Prometheus metrics exposed at `/metrics`:

- **Total Predictions** — `predictions_total`
- **Anomaly Rate** — `anomaly_rate`
- **Prediction Latency** — `rate(prediction_latency_seconds_sum[5m]) / rate(prediction_latency_seconds_count[5m])`
- **Predicted Energy Consumption** — `predicted_consumption_kwh`

## Kubernetes Configuration

- 2 replicas by default
- HorizontalPodAutoscaler scales 2→10 pods based on 70% CPU utilisation
- Liveness and readiness probes on `/health`
- Resource requests/limits configured for predictable scheduling

## Model Performance

Run `python train.py` to see current metrics. The Isolation Forest is configured with 5% contamination, and the XGBoost forecaster is evaluated with MAE and RMSE on a held-out 20% test split.

## Future Improvements

- Replace synthetic data with real ASHRAE Great Energy Predictor III dataset
- Add Grafana alerting rules for anomaly rate thresholds
- GitLab CI/CD pipeline for automated build and deploy
- Persist prediction history to a database for trend analysis
