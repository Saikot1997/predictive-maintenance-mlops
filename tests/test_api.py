"""FastAPI endpoints এর tests।"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """Test client তৈরি করো।"""
    with patch("src.api.main.redis_client", None):  # Redis mock
        from src.api.main import app
        with TestClient(app) as c:
            yield c


def test_root_endpoint(client):
    """Root endpoint কাজ করছে কি না।"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint(client):
    """Health check endpoint।"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_input(client):
    """Valid input দিয়ে prediction।"""
    with patch("src.api.main.predict") as mock_predict:
        mock_predict.return_value = {
            "prediction": 0,
            "probability": 0.05,
            "failure_type": "No Failure",
            "risk_level": "LOW",
            # main.py তে result["cached"] = False set করে PredictionResponse build করে।
            # predict() mock শুধু এই 4টা field return করলেই যথেষ্ট।
        }
        response = client.post("/predict", json={
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
            "type": "M",
        })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0, 1]


def test_predict_invalid_type(client):
    """Invalid machine type এ validation error।"""
    response = client.post("/predict", json={
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "type": "X",  # Invalid!
    })
    assert response.status_code == 422  # Validation error


def test_predict_out_of_range(client):
    """Out-of-range temperature validation।"""
    response = client.post("/predict", json={
        "air_temperature": 999.9,  # Way too high
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "type": "M",
    })
    assert response.status_code == 422