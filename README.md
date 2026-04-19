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

## 📖 About This Project

This project builds a **complete, production-ready MLOps pipeline** for predicting industrial
equipment failure using real sensor data. It is designed to demonstrate every layer of a
modern ML system — from raw data ingestion to real-time inference, monitoring, and CI/CD.

### What it does
- Ingests sensor readings from 5 simulated industrial machines via **Apache Kafka**
- Preprocesses and engineers 6 physics-based features from raw sensor data
- Trains **RandomForest** and **XGBoost** classifiers to predict machine failure (binary: fail / no fail)
- Tracks all experiments, metrics, and model versions in **MLflow** (PostgreSQL backend)
- Serves the best model ("champion") via a **FastAPI** REST API with **Redis** caching
- Monitors prediction metrics and API performance with **Prometheus** and **Grafana**
- Detects data drift between training and production distributions using **Evidently AI**
- Versions data and pipeline stages with **DVC** (Google Drive remote)
- Automates testing, linting, and Docker image publishing via **GitHub Actions CI/CD**

### Why this project
Predicting equipment failure before it happens saves factories from costly unplanned downtime.
This project uses the [AI4I 2020 dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)
— a realistic simulation of CNC machine sensor data with a 3.4% failure rate — to demonstrate
how a production ML system handles class imbalance, real-time streaming, caching, and drift monitoring.

### Key results
| Model | F1 Score | ROC-AUC |
|---|---|---|
| RandomForest | 0.748 | 0.978 |
| XGBoost (Champion) | **0.824** | **0.987** |

---

## ✨ Tech Stack

| Category | Tools |
|---|---|
| **ML Models** | scikit-learn (RandomForest), XGBoost |
| **Experiment Tracking** | MLflow 2.13 + PostgreSQL backend (Docker) |
| **Data Versioning** | DVC 3.50 + Google Drive remote |
| **Model Serving** | FastAPI + BentoML |
| **Stream Processing** | Apache Kafka + Zookeeper |
| **Caching** | Redis |
| **Monitoring** | Prometheus + Grafana + Evidently AI |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + Docker Compose (10 services) |
| **Code Quality** | pre-commit, black, flake8, isort, pytest |

---

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
│  MLflow experiment tracking (PostgreSQL backend)      │
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

## 🖥️ Prerequisites

### 1. Python 3.11

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### 2. Docker Engine + Docker Compose V2

```bash
# Remove old Docker versions if any
sudo apt remove docker docker-engine docker.io containerd runc 2>/dev/null

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Allow running docker without sudo (log out and back in after this)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version          # Docker version 24+
docker compose version    # Docker Compose version v2+
```

### 3. Git

```bash
sudo apt install -y git
```

### System Requirements
- OS: Ubuntu 22.04 / 24.04
- RAM: 8 GB minimum (Kafka + all services)
- Disk: 5 GB free

---

## 🚀 Complete Setup from Scratch

### Step 1 — Clone the repository

```bash
git clone https://github.com/Saikot1997/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops
```

### Step 2 — Create Python virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

> Takes 2–5 minutes on first install (~30 packages including MLflow, XGBoost, FastAPI, Evidently).

### Step 3 — Configure environment

```bash
cp .env.example .env
```

The defaults work for Docker out of the box. See the [Environment Variables](#️-environment-variables)
section if you want to run training scripts locally (outside Docker).

### Step 4 — Start all Docker services

```bash
docker compose up -d
```

Starts 10 services: PostgreSQL, MLflow, Redis, Zookeeper, Kafka, FastAPI,
Kafka Producer, Kafka Consumer, Prometheus, Grafana.

Wait ~35 seconds, then check everything is healthy:

```bash
docker compose ps
```

Expected output:
```
NAME            STATUS
pm_postgres     Up (healthy)   <- MLflow database
pm_redis        Up (healthy)   <- Prediction cache
pm_mlflow       Up (healthy)   <- Experiment tracking UI at :5000
pm_kafka        Up (healthy)   <- Message broker
pm_api          Up (healthy)   <- FastAPI at :8000
pm_zookeeper    Up             <- Required by Kafka
pm_producer     Up             <- Simulates machine sensor data
pm_consumer     Up             <- Reads stream, calls /predict
pm_prometheus   Up             <- Metrics collector at :9090
pm_grafana      Up             <- Dashboard at :3000
```

If `pm_mlflow` shows `unhealthy`, wait 20 more seconds — it waits for PostgreSQL to fully start.

### Step 5 — Run the ML pipeline

```bash
# Step 5a: Preprocess → generates data/processed/train.parquet and test.parquet
python -m src.data.preprocess

# Step 5b: Train models → registers best model in MLflow Registry
python -m src.models.train
```

Expected output:
```
RandomForest | F1: 0.7482 | ROC-AUC: 0.9780
XGBoost      | F1: 0.8235 | ROC-AUC: 0.9867
Best Model: XGBoost | f1_score: 0.8235
Model alias 'champion' set -> version 1
```

> The training script connects to MLflow at `http://localhost:5000` (the running Docker container).

### Step 6 — Verify end-to-end

```bash
# Health check (should show redis: true)
curl -s http://localhost:8000/health | python3 -m json.tool

# Normal prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":298.1,"process_temperature":308.6,"rotational_speed":1551,"torque":42.8,"tool_wear":0,"type":"M"}' \
  | python3 -m json.tool

# High-risk prediction
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"air_temperature":305.0,"process_temperature":318.0,"rotational_speed":1300,"torque":75.0,"tool_wear":250,"type":"L"}' \
  | python3 -m json.tool

# Prometheus
curl -s http://localhost:9090/-/ready

# MLflow
curl -s http://localhost:5000/health
```

Expected:
```json
{"prediction": 0, "probability": 0.05, "failure_type": "No Failure", "risk_level": "LOW", "cached": false}
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
| BentoML | http://localhost:3001 | requires --profile bentoml flag |

---

## 📁 Project Structure

```
predictive-maintenance-mlops/
|
├── .github/workflows/
|   ├── ci.yml                  <- Lint + test + drift check on every push
|   └── cd.yml                  <- Build + push Docker image on push to main
|
├── configs/
|   ├── data_config.yaml        <- Dataset paths, columns, split ratio
|   ├── model_config.yaml       <- RF and XGBoost hyperparameters
|   └── training_config.yaml    <- MLflow experiment name, drift thresholds
|
├── data/
|   ├── raw/predictive_maintenance.csv     <- AI4I 2020 dataset (10,000 rows)
|   ├── processed/
|   |   ├── train.parquet       <- Generated by preprocess.py (8000 rows)
|   |   └── test.parquet        <- Generated by preprocess.py (2000 rows)
|   └── reports/
|       ├── metrics.json        <- Evaluation metrics (F1, AUC, precision, recall)
|       └── drift_report_*.html <- Evidently reports (open in browser)
|
├── grafana/
|   ├── provisioning/           <- Auto-configures datasource + dashboard on startup
|   └── dashboards/mlops_dashboard.json  <- Pre-built Grafana dashboard
|
├── models/trained/
|   ├── xgboost.pkl             <- Trained model backup
|   ├── random_forest.pkl       <- Trained model backup
|   ├── scaler.pkl              <- Fitted StandardScaler (required for inference)
|   └── label_encoder.pkl       <- Fitted LabelEncoder for Type column (L/M/H)
|
├── prometheus/prometheus.yml   <- Scrapes FastAPI /metrics every 15 seconds
|
├── src/
|   ├── api/
|   |   ├── main.py             <- FastAPI app (predict, health, metrics endpoints)
|   |   ├── schemas.py          <- Pydantic v2 request/response validation
|   |   └── bento_service.py    <- BentoML service definition
|   ├── data/
|   |   ├── preprocess.py       <- Full preprocessing pipeline
|   |   └── download_data.py    <- Kaggle API downloader (optional)
|   ├── features/
|   |   └── feature_engineering.py  <- 6 derived physics-based features
|   ├── models/
|   |   ├── train.py            <- RF + XGBoost training with MLflow logging
|   |   ├── evaluate.py         <- Loads champion model, computes metrics
|   |   └── predict.py          <- Singleton model loader + inference
|   ├── monitoring/
|   |   └── data_drift.py       <- Evidently drift report generator
|   └── streaming/
|       ├── kafka_producer.py   <- Simulates 5 machines at 1 msg/sec, 3% failure rate
|       └── kafka_consumer.py   <- Reads Kafka stream -> calls /predict -> logs HIGH RISK
|
├── tests/
|   ├── test_api.py             <- 5 FastAPI endpoint tests
|   ├── test_data.py            <- 4 preprocessing tests
|   └── test_features.py        <- 4 feature engineering tests
|
├── .env                        <- Your local config (not committed to git)
├── .env.example                <- Template -- copy to .env on setup
├── .pre-commit-config.yaml     <- black, isort, flake8, yaml/json checks
├── docker-compose.yml          <- All 10 services
├── Dockerfile                  <- FastAPI image (python:3.11-slim, non-root user)
├── Dockerfile.mlflow           <- MLflow image (PostgreSQL backend)
├── dvc.yaml                    <- DVC pipeline stages
├── params.yaml                 <- Hyperparameters tracked by DVC
├── Makefile                    <- Shortcut commands
├── requirements.txt            <- Production dependencies (pinned versions)
└── requirements-dev.txt        <- Dev dependencies (pytest, black, flake8, isort)
```

---

## 🔬 ML Pipeline Details

### Dataset
**AI4I 2020 Predictive Maintenance** — 10,000 rows, 14 columns
- Sensor readings: air temperature (K), process temperature (K), rotational speed (RPM), torque (Nm), tool wear (min)
- Machine type: L / M / H
- Target: binary machine failure (3.4% positive rate — class imbalanced)

### Feature Engineering (6 derived features)
| Feature | Formula | Rationale |
|---|---|---|
| `temp_diff` | process_temp - air_temp | High diff = cooling problem |
| `temp_ratio` | process_temp / air_temp | Normalized thermal load |
| `power_consumption` | torque x rpm x 2pi/60 | Physics-based power (Watts) |
| `power_wear_interaction` | power x tool_wear | Combined stress indicator |
| `wear_speed_ratio` | tool_wear / rpm | Wear rate per unit speed |
| `torque_per_wear` | torque / (tool_wear + 1) | Stress per wear unit |

### Preprocessing Order (no data leakage)
```
Raw CSV
  -> clean (drop UDI/ProductID, fill NaN with median/mode, remove duplicates)
  -> encode Type column (LabelEncoder: H=0, L=1, M=2) -- fit on full dataset
  -> rename columns to snake_case
  -> feature engineering on RAW sensor values (before scaling)
  -> train/test split (80/20, stratified on target)
  -> StandardScaler fit on X_train ONLY -> transform X_train and X_test
  -> save train.parquet (8000 rows) + test.parquet (2000 rows)
```

### Training
- **RandomForest**: 200 trees, max_depth=15, class_weight="balanced"
- **XGBoost**: 300 estimators, scale_pos_weight=30, eval_metric=aucpr
- Best model by F1-score registered in MLflow Registry with alias `champion`
- Local `.pkl` backups in `models/trained/` as fallback

---

## ⚙️ Environment Variables

### Docker vs Local

Inside Docker containers, services communicate by container name:
```
MLFLOW_TRACKING_URI=http://mlflow:5000
REDIS_HOST=redis
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

Local Python scripts connect to Docker via localhost ports:
```
MLFLOW_TRACKING_URI=http://localhost:5000
REDIS_HOST=localhost
KAFKA_BOOTSTRAP_SERVERS=localhost:9093
```

`docker-compose.yml` overrides the critical vars per service, so the `.env` file
is mainly used when running scripts locally. The `.env.example` defaults work fine
for Docker-only usage.

Full reference:
```bash
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow_password
POSTGRES_DB=mlflow
POSTGRES_HOST=localhost          # use 'postgres' inside Docker
POSTGRES_PORT=5432

MLFLOW_TRACKING_URI=http://localhost:5000   # use 'http://mlflow:5000' inside Docker
MLFLOW_EXPERIMENT_NAME=predictive_maintenance_v1

REDIS_HOST=localhost             # use 'redis' inside Docker
REDIS_PORT=6379
REDIS_PASSWORD=                  # leave empty (no auth configured)

KAFKA_BOOTSTRAP_SERVERS=localhost:9093   # use 'kafka:9092' inside Docker
KAFKA_TOPIC_SENSOR=sensor-data
KAFKA_TOPIC_PREDICTIONS=predictions

API_HOST=0.0.0.0
API_PORT=8000
MODEL_NAME=predictive_maintenance_model

KAGGLE_USERNAME=your_kaggle_username   # only needed to download dataset manually
KAGGLE_KEY=your_kaggle_api_key
```

---

## 🔧 Development Commands

```bash
make setup          # Create venv + install all deps + pre-commit hooks
make install        # pip install only (no venv creation)
make lint           # black --check + isort --check + flake8
make format         # Auto-fix formatting with black + isort
make test           # pytest tests/ -v --cov=src --cov-report=term-missing
make download       # Download dataset from Kaggle (needs credentials in .env)
make train          # python -m src.models.train
make pipeline       # dvc repro (preprocess -> train -> evaluate -> drift_check)
make docker-up      # docker compose up --build -d
make docker-down    # docker compose down (volumes preserved)
make clean-volumes  # docker compose down -v (deletes all data -- fresh start)
make drift          # python -m src.monitoring.data_drift
make logs           # docker compose logs -f api
make mlflow-ui      # Open http://localhost:5000 in browser
make clean          # Remove .pyc, __pycache__, .pytest_cache
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root — API name and link to docs |
| GET | `/health` | Health check including Redis status |
| POST | `/predict` | Predict machine failure from sensor readings |
| GET | `/metrics` | Prometheus metrics endpoint |
| GET | `/model/info` | Loaded model name and registry stage |
| GET | `/docs` | Interactive Swagger UI |

**Request schema** (values outside ranges → HTTP 422):
```json
{
  "air_temperature": 298.1,       // 290.0 to 310.0 K
  "process_temperature": 308.6,   // 300.0 to 320.0 K
  "rotational_speed": 1551,       // 1000.0 to 3000.0 RPM
  "torque": 42.8,                 // 0.0 to 100.0 Nm
  "tool_wear": 0,                 // 0.0 to 300.0 min
  "type": "M"                     // "L", "M", or "H"
}
```

**Response schema:**
```json
{
  "prediction": 0,                // 0 = No Failure, 1 = Machine Failure
  "probability": 0.0195,          // 0.0 to 1.0
  "failure_type": "No Failure",
  "risk_level": "LOW",            // LOW (<0.3) | MEDIUM (0.3-0.7) | HIGH (>0.7)
  "cached": false                 // true if served from Redis cache
}
```

---

## 📊 Monitoring

### Grafana (http://localhost:3000 — admin/admin)
Dashboard is auto-provisioned on startup. Panels:
- Total predictions counter
- HIGH risk predictions counter (red alert)
- Request rate over time
- P95 API latency
- Redis cache hit rate

### Prometheus (http://localhost:9090)
Scrapes `http://api:8000/metrics` every 15s. Key metrics:
- `predictions_total{risk_level="HIGH"}`
- `request_duration_seconds`
- `cache_hits_total` / `cache_misses_total`
- `api_requests_total`

### Evidently Data Drift
```bash
source venv/bin/activate
python -m src.monitoring.data_drift
xdg-open data/reports/drift_report_*.html
```
Compares training vs test distributions. Alerts if >10% of features drift significantly.

---

## 🌊 Kafka Streaming

Runs automatically inside Docker. To monitor:
```bash
docker logs pm_producer -f   # see sensor readings being produced
docker logs pm_consumer -f   # see predictions and HIGH RISK alerts
```

The consumer reconnects automatically if the API restarts.

---

## 🧪 Tests

```bash
source venv/bin/activate
pytest tests/ -v --cov=src --cov-report=term-missing
```

13 tests, all passing. No Docker needed — Redis and MLflow are mocked.

| File | Count | What it covers |
|---|---|---|
| `test_api.py` | 5 | Root, health, valid prediction, invalid type (422), out-of-range (422) |
| `test_data.py` | 4 | Duplicate removal, missing values, Type encoding, row count |
| `test_features.py` | 4 | Temperature features, temp_diff formula, power features, no NaN |

---

## 🔄 DVC

```bash
dvc repro          # Run full pipeline (skips unchanged stages)
dvc metrics show   # Print metrics.json
dvc dag            # Show pipeline dependency graph
dvc pull           # Download data from Google Drive (OAuth prompt on first use)
dvc push           # Upload data to Google Drive
dvc params diff    # Compare hyperparameters between git commits
```

Pipeline: `preprocess -> train -> evaluate -> drift_check`

---

## 🚀 CI/CD

**CI** (every push): lint -> test -> Docker build check -> Evidently drift check

**CD** (push to main): builds and pushes Docker image to Docker Hub

Required GitHub Secrets (Settings -> Secrets -> Actions):
```
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
```

---

## 🍱 BentoML (Optional)

```bash
# First: save the trained model to BentoML store
source venv/bin/activate
python -c "from src.api.bento_service import save_model_to_bentoml; save_model_to_bentoml()"

# Start BentoML on port 3001
docker compose --profile bentoml up -d bentoml
```

---

## 🔁 After a Reboot

```bash
cd ~/Projects/predictive-maintenance-mlops
source venv/bin/activate
docker compose up -d
sleep 60 && docker compose ps
curl -s http://localhost:8000/health
```

No retraining needed — models persist in PostgreSQL (MLflow) and `models/trained/*.pkl`.

---

## 🐳 Docker Tips

| Situation | Command |
|---|---|
| Code change, no new packages | `docker cp src/models/predict.py pm_api:/app/src/models/predict.py && docker compose restart api` |
| New package in requirements.txt | `docker compose up -d --build api` |
| View API logs live | `docker compose logs -f api` |
| Stop everything, keep data | `docker compose down` |
| Full reset, delete all data | `docker compose down -v && docker compose up -d` |
| Check resource usage | `docker stats` |

---

## 🛠️ Troubleshooting

**`pm_mlflow` unhealthy**
Wait 30s. PostgreSQL may still be starting. Then: `docker compose restart mlflow`

**API returns 500 on predictions**
Models not found. Run training first, then check:
```bash
docker exec pm_api ls /app/models/trained/
# Must show: xgboost.pkl, scaler.pkl, label_encoder.pkl
```

**Kafka consumer connection errors on startup**
Normal — consumer starts before Kafka is ready and retries automatically. Wait 30s.

**`dvc pull` asks for browser auth**
Expected on first use. Approve Google OAuth and credentials are cached locally.

**`python -m src.models.train` cannot reach MLflow**
Docker must be running: `docker compose ps | grep mlflow`

**Port already in use**
`docker compose down && docker compose up -d`

**Pre-commit hook fails on commit**
`make format && git add -A && git commit -m "your message"`

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with Python 3.11 · XGBoost · FastAPI · MLflow · DVC · Kafka · Redis · Prometheus · Grafana · Docker*
