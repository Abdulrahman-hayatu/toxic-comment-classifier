"""
Tests for the API endpoints.
These tests use FastAPI's TestClient to simulate requests to the API without needing to run a live server.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

# Mock the classifier so tests don't need actual models
mock_prediction = MagicMock()
mock_prediction.labels = {l: 0 for l in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']}
mock_prediction.probabilities = {l: 0.1 for l in mock_prediction.labels}
mock_prediction.uncertainty = {l: 0.02 for l in mock_prediction.labels}
mock_prediction.risk_tier = "CLEAN"
mock_prediction.flagged_labels = []
mock_prediction.requires_review = False


@pytest.fixture
def client():
    with patch('api.model_loader.classifier', MagicMock()) as mock_clf:
        mock_clf.predict.return_value = mock_prediction
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_text(client):
    response = client.post("/predict", json={"text": "Hello world"})
    assert response.status_code == 200
    data = response.json()
    assert "risk_tier" in data
    assert "flagged_labels" in data
    assert "predictions" in data


def test_predict_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Validation error


def test_predict_too_long_text_rejected(client):
    response = client.post("/predict", json={"text": "x" * 5001})
    assert response.status_code == 422