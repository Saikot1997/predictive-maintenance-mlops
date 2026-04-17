.PHONY: help setup install lint format test train pipeline docker-up docker-down clean-volumes logs mlflow-ui drift clean

# Default: help দেখাও
help:
	@echo "Predictive Maintenance MLOps — Available Commands"
	@echo "─────────────────────────────────────────────────"
	@echo "  make setup        → Virtual env তৈরি ও dependencies install"
	@echo "  make install      → Dependencies install"
	@echo "  make lint         → Code quality check"
	@echo "  make test         → Tests run করো"
	@echo "  make download     → Kaggle dataset download"
	@echo "  make train        → Model training চালাও (MLflow tracking সহ)"
	@echo "  make pipeline     → DVC pipeline চালাও (preprocess → train → evaluate)"
	@echo "  make docker-up      → সব Docker services start করো"
	@echo "  make docker-down    → সব Docker services stop করো (volumes preserved)"
	@echo "  make clean-volumes  → Services stop ও সব volumes delete করো (fresh start)"
	@echo "  make mlflow-ui    → MLflow UI open করো"
	@echo "  make drift        → Data drift check করো"
	@echo "  make clean        → Build artifacts মুছে ফেলো"

setup:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt
	. venv/bin/activate && pre-commit install
	cp .env.example .env
	@echo "✅ Setup complete! 'source venv/bin/activate' দিয়ে activate করো।"

install:
	pip install -r requirements.txt -r requirements-dev.txt

lint:
	black --check src/ tests/
	isort --check-only --profile black src/ tests/
	flake8 src/ tests/ --max-line-length=100

format:
	black src/ tests/
	isort --profile black src/ tests/

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

download:
	python -m src.data.download_data

train:
	python -m src.models.train

pipeline:
	@echo "Note: data/raw/predictive_maintenance.csv না থাকলে আগে 'dvc pull' দাও"
	dvc repro
	dvc metrics show

docker-up:
	docker compose up --build -d
	@echo "✅ All services started!"
	@echo "  FastAPI:    http://localhost:8000/docs"
	@echo "  MLflow:     http://localhost:5000"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"

docker-down:
	docker compose down
	@echo "✅ All services stopped. (Data volumes preserved)"
	@echo "  To also delete all volumes: make clean-volumes"

clean-volumes:
	docker compose down -v
	@echo "✅ All services stopped and volumes deleted (fresh state)."

logs:
	docker compose logs -f api

mlflow-ui:
	@echo "MLflow UI → http://localhost:5000"
	@xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || true

drift:
	python -m src.monitoring.data_drift

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null; true
	rm -rf htmlcov/ .coverage
	@echo "✅ Cleaned!"