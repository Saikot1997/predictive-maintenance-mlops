# Base image: Python 3.11 slim (ছোট size)
FROM python:3.11-slim

# Metadata
LABEL maintainer="Your Name"
LABEL description="Predictive Maintenance FastAPI Service"

# Non-root user তৈরি করো (security best practice)
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Work directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies আগে copy করো (Docker layer cache optimize)
# এটা করলে code বদলালে dependencies আবার install হবে না
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application code copy করো
COPY src/ ./src/
COPY configs/ ./configs/
# NOTE: models/ folder copy করা হচ্ছে না কারণ:
# 1. .gitignore তে আছে (DVC manage করে)
# 2. Build time এ folder খালি থাকলে API startup এ crash করবে
# 3. docker-compose.yml এ volume mount করা আছে: ./models:/app/models
# models/ folder DVC দিয়ে: dvc pull করার পরে docker compose up দাও
RUN mkdir -p /app/models/trained
COPY .env.example .env

# Ownership দাও
RUN chown -R appuser:appuser /app

# Non-root user তে switch করো
USER appuser

# Port expose করো
EXPOSE 8000

# Health check — Docker জানবে container healthy কি না
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# App start করো
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--log-level", "info"]