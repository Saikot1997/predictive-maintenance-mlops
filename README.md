# 🏭 Predictive Maintenance MLOps Pipeline

[![CI](https://github.com/Saikot1997/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Saikot1997/predictive-maintenance-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/Saikot1997/predictive-maintenance-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/Saikot1997/predictive-maintenance-mlops/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker)](https://docker.com)
[![MLflow](https://img.shields.io/badge/mlflow-2.13-0194E2.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/dvc-3.50-945DD6.svg)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade MLOps pipeline** that predicts industrial equipment failure before it happens,
built on the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

**Model performance (XGBoost Champion):** F1 = **0.824** | ROC-AUC = **0.987**

---

## ✨ Tech Stack

| Category | Tools |
|---|---|
| **ML Models** | scikit-learn (RandomForest), XGBoost |
| **Experiment Tracking** | MLflow 2.13 + SQLite backend |
| **Data Versioning** | DVC 3.50 + Google Drive remote |
| **Model Serving** | FastAPI + BentoML |
| **Stream Processing** | Apache Kafka + Zookeeper |
| **Caching** | Redis |
| **Monitoring** | Prometheus + Grafana + Evidently AI |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + Docker Compose (10 services) |
| **Code Quality** | pre-commit, black, flake8, isort, pytest |

---

┌──────────────────────────────────────────────────────────────┐
│                        PROJECT STATUS                         │
├──────────────────────────────────────────────────────────────┤
│  This project is under development.                           │
│  Bug fixing is ongoing and improvements are in progress.      │
└──────────────────────────────────────────────────────────────┘

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                     DATA LAYER                        │
│  Kaggle CSV ──▶ DVC (versioning) ──▶ Google Drive    │
│  Apache Kafka (real-time sensor stream)               │
└─────────────────────────┬────────────────────────────┘
                           │
┌─────────────────────────▼────────────────────────────┐
│                   PIPELINE LAYER                      │
│  preprocess.py ──▶ feature_engineering.py             │
│  DVC pipeline orchestrates all steps                  │
└─────────────────────────┬────────────────────────────┘
                           │
┌─────────────────────────▼────────────────────────────┐
│                    MODEL LAYER                        │
│  RandomForest + XGBoost training                      │
│  MLflow experiment tracking (SQLite backend)          │
│  MLflow Model Registry (champion alias)               │
└─────────────────────────┬────────────────────────────┘
                           │
┌─────────────────────────▼────────────────────────────┐
│                   SERVING LAYER                       │
│  FastAPI :8000 ──▶ Redis cache ──▶ predict.py         │
│  BentoML :3001 (optional ML-native serving)           │
└─────────────────────────┬────────────────────────────┘
                           │
┌─────────────────────────▼────────────────────────────┐
│                  MONITORING LAYER                     │
│  Prometheus :9090 ──▶ Grafana :3000                   │
│  Evidently AI (data drift HTML reports)               │
└──────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 22.04 / 24.04
- Docker Engine 24+ and Docker Compose V2
- Python 3.11
- Git

### Step 1 — Clone and set up environment

```bash
git clone https://github.com/Saikot1997/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Step 2 — Start Docker services

```bash
docker compose up -d
```

Wait ~30 seconds for all services to become healthy:

```bash
docker compose ps
```

### Step 3 — Start local MLflow server

```bash
mkdir -p mlartifacts
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 &
sleep 5
curl -s http://localhost:5000/health && echo "MLflow is UP"
```

### Step 4 — Run the ML pipeline

```bash
# Preprocess raw data
python -m src.data.preprocess

# Train RandomForest + XGBoost, register best model
python -m src.models.train
```

Expected training output:
```
RandomForest | F1: 0.7482 | ROC-AUC: 0.9780
XGBoost      | F1: 0.8235 | ROC-AUC: 0.9867
🏆 Best Model: XGBoost | f1_score: 0.8235
✅ Model alias 'champion' set → version 3
```

### Step 5 — Test a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature": 298.1,
    "process_temperature": 308.6,
    "rotational_speed": 1551,
    "torque": 42.8,
    "tool_wear": 0,
    "type": "M"
  }'
```

Expected response:
```json
{"prediction": 0, "probability": 0.0195, "failure_type": "No Failure", "risk_level": "LOW", "cached": false}
```

High-risk scenario (tool_wear=250, high torque):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":305.0,"process_temperature":318.0,"rotational_speed":1300,"torque":75.0,"tool_wear":250,"type":"L"}'
```
```json
{"prediction": 1, "probability": 0.9997, "failure_type": "Machine Failure", "risk_level": "HIGH", "cached": false}
```

---

## 📍 Service URLs

| Service | URL | Credentials |
|---|---|---|
| FastAPI Swagger UI | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Grafana Dashboard | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| BentoML | http://localhost:3001 | `docker compose --profile bentoml up bentoml` |

---

## 📁 Project Structure

```
predictive-maintenance-mlops/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                  ← Lint + test on every push
│       └── cd.yml                  ← Build + push Docker image on main
│
├── configs/
│   ├── data_config.yaml            ← Dataset paths, columns, split ratio
│   ├── model_config.yaml           ← RF & XGBoost hyperparameters
│   └── training_config.yaml        ← MLflow experiment config, monitoring thresholds
│
├── data/
│   ├── raw/
│   │   ├── predictive_maintenance.csv          ← AI4I 2020 dataset (DVC tracked)
│   │   └── predictive_maintenance.csv.dvc      ← DVC pointer file
│   ├── processed/
│   │   ├── train.parquet           ← Preprocessed training set (8000 rows)
│   │   └── test.parquet            ← Preprocessed test set (2000 rows)
│   └── reports/
│       ├── metrics.json            ← Evaluation metrics (DVC tracked)
│       └── drift_report_*.html     ← Evidently drift reports
│
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/dashboard.yml
│   │   └── datasources/prometheus.yml
│   └── dashboards/
│       └── mlops_dashboard.json    ← Pre-built Grafana dashboard
│
├── models/
│   └── trained/
│       ├── random_forest.pkl       ← Trained RF model (local backup)
│       ├── xgboost.pkl             ← Trained XGBoost model (local backup)
│       ├── scaler.pkl              ← Fitted StandardScaler
│       └── label_encoder.pkl       ← Fitted LabelEncoder for 'Type' column
│
├── prometheus/
│   └── prometheus.yml              ← Scrape config (FastAPI /metrics every 15s)
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                 ← FastAPI app (predict, health, metrics endpoints)
│   │   ├── schemas.py              ← Pydantic request/response schemas
│   │   └── bento_service.py        ← BentoML 1.x service
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py           ← Clean → encode → feature engineer → scale → split
│   │   └── download_data.py        ← Kaggle API dataset downloader
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  ← Temperature, power, wear derived features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                ← RF + XGBoost training with MLflow logging
│   │   ├── evaluate.py             ← Load champion model, compute metrics, save JSON
│   │   └── predict.py              ← Singleton model loader + inference logic
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── data_drift.py           ← Evidently drift report generator
│   └── streaming/
│       ├── __init__.py
│       ├── kafka_producer.py        ← Simulates 5 machines, 3% failure rate
│       └── kafka_consumer.py        ← Reads sensor stream → calls /predict API
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 ← FastAPI endpoint tests (5 tests)
│   ├── test_data.py                ← Preprocessing tests (4 tests)
│   └── test_features.py            ← Feature engineering tests (4 tests)
│
├── mlartifacts/                    ← MLflow artifact store (local)
├── mlflow.db                       ← MLflow SQLite backend
│
├── .env                            ← Local secrets (not committed)
├── .env.example                    ← Template for .env
├── .gitignore
├── .pre-commit-config.yaml         ← black, isort, flake8, trailing-whitespace
├── docker-compose.yml              ← 10 services: api, mlflow, postgres, redis,
│                                       kafka, zookeeper, producer, consumer,
│                                       prometheus, grafana
├── Dockerfile                      ← FastAPI app image (python:3.11-slim, non-root)
├── Dockerfile.mlflow               ← MLflow server image with PostgreSQL client
├── dvc.yaml                        ← DVC pipeline: preprocess → train → evaluate
├── params.yaml                     ← DVC-tracked hyperparameters
├── Makefile                        ← Developer shortcuts (see below)
├── requirements.txt                ← Production dependencies (pinned versions)
└── requirements-dev.txt            ← Dev dependencies (pytest, black, flake8, isort)
```

---

## 🔬 ML Pipeline Details

### Dataset
**AI4I 2020 Predictive Maintenance** — 10,000 rows, 14 columns
- Sensor readings: air temperature (K), process temperature (K), rotational speed (RPM), torque (Nm), tool wear (min)
- Machine type: L / M / H (Low / Medium / High quality)
- Target: binary machine failure (3.4% positive rate — imbalanced)
- Failure subtypes: TWF, HDF, PWF, OSF, RNF

### Feature Engineering (6 derived features)
| Feature | Formula | Rationale |
|---|---|---|
| `temp_diff` | process_temp − air_temp | High diff = cooling problem |
| `temp_ratio` | process_temp / air_temp | Normalized thermal load |
| `power_consumption` | torque × rpm × 2π/60 | Physics-based power (Watts) |
| `power_wear_interaction` | power × tool_wear | Combined stress indicator |
| `wear_speed_ratio` | tool_wear / rpm | Wear rate per unit speed |
| `torque_per_wear` | torque / (tool_wear + 1) | Stress per wear unit |

### Preprocessing Pipeline (correct order, no data leakage)
```
Raw CSV
  → clean (drop UDI/ProductID, fill NaN with median/mode)
  → encode Type (LabelEncoder: H=0, L=1, M=2)
  → rename columns to snake_case
  → feature engineering on RAW values (physics formulas need real units)
  → train/test split (80/20, stratified)
  → StandardScaler fit on X_train only → transform X_train and X_test
  → save train.parquet and test.parquet
```

### Training
- **RandomForest**: 200 trees, max_depth=15, class_weight="balanced", sample_weight computed per-class
- **XGBoost**: 300 estimators, scale_pos_weight=30 (handles 3.4% minority class), eval_metric=aucpr
- Best model selected by F1-score → registered in MLflow Model Registry with alias `champion`

---

## 🔧 Development Commands (Makefile)

```bash
make setup          # Create venv + install deps + pre-commit hooks
make install        # Install requirements only
make lint           # black --check + isort --check + flake8
make format         # Auto-format with black + isort
make test           # pytest tests/ -v --cov=src
make download       # Download Kaggle dataset
make train          # Run model training
make pipeline       # Run full DVC pipeline (preprocess → train → evaluate)
make docker-up      # docker compose up --build -d
make docker-down    # docker compose down (volumes preserved)
make clean-volumes  # docker compose down -v (fresh state)
make drift          # Run Evidently data drift report
make logs           # docker compose logs -f api
make mlflow-ui      # Open MLflow UI in browser
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root — API info |
| GET | `/health` | Health check (includes Redis status) |
| POST | `/predict` | Predict machine failure from sensor readings |
| GET | `/metrics` | Prometheus metrics (scraped every 15s) |
| GET | `/model/info` | Current production model name and stage |
| GET | `/docs` | Swagger UI (auto-generated) |

### Prediction Request Schema
```json
{
  "air_temperature": 298.1,        // float, 290–310 K
  "process_temperature": 308.6,    // float, 300–320 K
  "rotational_speed": 1551,        // float, 1000–3000 RPM
  "torque": 42.8,                  // float, 0–100 Nm
  "tool_wear": 0,                  // float, 0–300 min
  "type": "M"                      // "L", "M", or "H"
}
```

### Prediction Response Schema
```json
{
  "prediction": 0,                 // 0 = No Failure, 1 = Failure
  "probability": 0.0195,           // failure probability (0.0–1.0)
  "failure_type": "No Failure",    // human-readable label
  "risk_level": "LOW",             // LOW / MEDIUM / HIGH
  "cached": false                  // true if served from Redis cache
}
```

---

## 📊 Monitoring

### Prometheus Metrics (exposed at `/metrics`)
| Metric | Type | Description |
|---|---|---|
| `api_requests_total` | Counter | Total requests by method, endpoint, status |
| `predictions_total` | Counter | Predictions by result and risk level |
| `request_duration_seconds` | Histogram | Latency per endpoint |
| `cache_hits_total` | Counter | Redis cache hits |
| `cache_misses_total` | Counter | Redis cache misses |

### Grafana Dashboard (auto-provisioned)
- Total predictions (stat panel)
- HIGH risk predictions (stat panel, red threshold)
- Request rate over time (time series)
- API P95 latency (time series)
- Cache hit rate (stat panel)

### Data Drift (Evidently AI)
```bash
python -m src.monitoring.data_drift
# Generates: data/reports/drift_report_YYYYMMDD_HHMM.html
#            data/reports/target_drift_YYYYMMDD_HHMM.html
```
Compares training distribution vs test/production distribution using KS test (numeric) and chi-squared (categorical). Alerts if >10% of columns drift significantly.

---

## 🌊 Kafka Streaming

The producer simulates 5 machines sending sensor readings every second with a realistic 3% failure rate:

```bash
# Run locally (outside Docker)
python -m src.streaming.kafka_producer

# Consumer reads stream and calls /predict for each reading
python -m src.streaming.kafka_consumer
```

Both run automatically as Docker services (`pm_producer`, `pm_consumer`).

---

## 🧪 Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

**13 tests, all passing:**
- `test_api.py` — root, health, valid prediction, invalid type, out-of-range validation
- `test_data.py` — duplicate removal, missing value handling, encoding, shape preservation
- `test_features.py` — temperature features, power features, NaN-free output

---

## 🔄 DVC Pipeline

```bash
dvc repro          # Run all stages (skips unchanged stages)
dvc metrics show   # Show evaluation metrics
dvc dag            # Visualize pipeline as ASCII DAG
dvc params diff    # Compare params between commits
```

Pipeline stages (`dvc.yaml`):
```
preprocess → train → evaluate
```

---

## 🚀 CI/CD (GitHub Actions)

**CI** (`.github/workflows/ci.yml`) — triggers on every push:
1. Install dependencies
2. `black --check` (format)
3. `isort --check` (import order)
4. `flake8` (lint)
5. `pytest --cov` (tests + coverage)
6. Docker build verification (no push)

**CD** (`.github/workflows/cd.yml`) — triggers on push to `main`:
1. Login to Docker Hub
2. Build and push image with git SHA tag
3. Post deployment summary

Add these secrets to your GitHub repo (Settings → Secrets → Actions):
```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

---

## ⚙️ Environment Variables (.env)

```bash
# PostgreSQL (used by Docker MLflow container — not local MLflow)
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password
POSTGRES_DB=mlflow
POSTGRES_HOST=localhost
POSTGRES_PORT=5432

# MLflow — points to local MLflow server
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=predictive_maintenance_v1

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Kafka (use 9093 for external/local access, 9092 for inter-container)
KAFKA_BOOTSTRAP_SERVERS=localhost:9093
KAFKA_TOPIC_SENSOR=sensor-data
KAFKA_TOPIC_PREDICTIONS=predictions

# API
API_HOST=0.0.0.0
API_PORT=8000
MODEL_NAME=predictive_maintenance_model
```

---

## 🔁 Starting the Project After a Reboot

```bash
cd ~/Projects/predictive-maintenance-mlops
source venv/bin/activate

# 1. Start all Docker services
docker compose up -d

# 2. Start local MLflow (must be done manually)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 --port 5000 &

# 3. Verify everything is up
curl -s http://localhost:8000/health    # {"status":"healthy","redis":true}
curl -s http://localhost:5000/health   # OK
curl -s http://localhost:9090/-/ready  # Prometheus Server is Ready.
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Python 3.11 · XGBoost · FastAPI · MLflow · DVC · Kafka · Redis · Prometheus · Grafana · Docker*
