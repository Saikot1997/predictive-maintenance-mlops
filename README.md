# 🏭 Predictive Maintenance MLOps Pipeline

[![CI](https://github.com/YOUR_USERNAME/predictive-maintenance-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/predictive-maintenance-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/YOUR_USERNAME/predictive-maintenance-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/predictive-maintenance-mlops/actions/workflows/cd.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-compose-2496ED.svg?logo=docker)](https://docker.com)
[![MLflow](https://img.shields.io/badge/mlflow-2.13-0194E2.svg)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/dvc-3.50-945DD6.svg)](https://dvc.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade MLOps pipeline** that predicts industrial equipment failure before it happens.  
Built on the [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020).

---

## ✨ Features

| Category | Tools |
|---|---|
| **ML Framework** | scikit-learn, XGBoost |
| **Experiment Tracking** | MLflow + PostgreSQL backend |
| **Data Versioning** | DVC + Google Drive remote |
| **Model Serving** | FastAPI + BentoML |
| **Stream Processing** | Apache Kafka + Zookeeper |
| **Caching** | Redis |
| **Monitoring** | Prometheus + Grafana + Evidently AI |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + Docker Compose |
| **Code Quality** | pre-commit, black, flake8, isort, pytest |

---

## 🏗️ Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│ Kaggle Data │────▶│  DVC Track  │────▶│  Google Drive    │
└─────────────┘     └─────────────┘     └──────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   DVC Pipeline         │
              │  preprocess → features │
              │       → train          │
              └────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   MLflow    │
                    │ + PostgreSQL│
                    └──────┬──────┘
                           │
           ┌───────────────┼────────────────┐
           ▼               ▼                ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │ FastAPI │    │ BentoML  │    │  Redis   │
      │  :8000  │    │  :3001   │    │ (Cache)  │
      └────┬────┘    └──────────┘    └──────────┘
           │
    ┌──────┴───────┐
    │    Kafka     │
    │  Streaming   │
    └──────┬───────┘
           │
    ┌──────┴────────────────┐
    ▼                       ▼
┌──────────┐         ┌──────────────┐
│Prometheus│────────▶│   Grafana    │
│  :9090   │         │    :3000     │
└──────────┘         └──────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Engine 24+
- Docker Compose V2
- Git

### 3-Command Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/predictive-maintenance-mlops.git
cd predictive-maintenance-mlops

# 2. Environment
cp .env.example .env

# 3. Launch everything
docker compose up --build
```

### 📍 Service URLs

| Service | URL | Credentials |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Grafana | http://localhost:3000 | admin / admin |
| Prometheus | http://localhost:9090 | — |
| BentoML | http://localhost:3001 | — (চালাতে: `docker compose --profile bentoml up bentoml`) |

### Test a Prediction

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

---

## 🔧 Development Setup

```bash
# Virtual environment
python3 -m venv venv && source venv/bin/activate

# Dependencies
make install

# Pre-commit hooks
pre-commit install

# Run tests
make test

# Full ML pipeline
make pipeline
```

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance** — 10,000 data points, 14 features  
Sensor readings from a manufacturing system: air/process temperature, rotational speed, torque, tool wear.  
**Target**: Binary machine failure (3.4% failure rate — imbalanced dataset).

---

## 🔬 ML Pipeline

```
Raw CSV → Preprocessing → Feature Engineering → Train (RF + XGBoost)
       ↓ DVC tracks                              ↓ MLflow logs
       Google Drive                          Model Registry (Production)
```

**Models**: RandomForest + XGBoost  
**Handling imbalance**: class_weight="balanced" + scale_pos_weight  
**Best model selection**: F1-score on test set

---

## 📡 Monitoring

- **Data Drift**: Evidently AI generates HTML reports comparing training vs production data
- **System Metrics**: Prometheus scrapes `/metrics` → Grafana dashboards
- **Kafka Stream**: Real-time sensor simulation → live predictions

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src --cov-report=html
```

Coverage includes: data preprocessing, feature engineering, API endpoints, schemas.

---

## 📁 Project Structure

```
predictive-maintenance-mlops/
├── .github/workflows/    ← CI/CD pipelines
├── configs/              ← YAML configuration files
├── data/                 ← DVC-tracked datasets
├── grafana/              ← Dashboard provisioning
├── models/               ← Trained model artifacts
├── prometheus/           ← Metrics config
├── src/
│   ├── api/              ← FastAPI + BentoML serving
│   ├── data/             ← Data loading & preprocessing
│   ├── features/         ← Feature engineering
│   ├── models/           ← Training, evaluation, prediction
│   ├── monitoring/       ← Evidently + Prometheus metrics
│   └── streaming/        ← Kafka producer & consumer
├── tests/                ← pytest test suite
├── docker-compose.yml    ← Full stack orchestration
├── dvc.yaml              ← ML pipeline definition
└── Makefile              ← Developer shortcuts
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built with ❤️ | MLOps Best Practices | Production-Ready*