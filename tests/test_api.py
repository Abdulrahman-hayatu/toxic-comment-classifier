"""
Unit tests for the API endpoints using FastAPI's TestClient.
This file should be run with pytest and assumes that the API's model loading is mocked
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Dict, List

# Build a realistic mock Prediction object matching the dataclass shape
LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

mock_prediction = MagicMock()
mock_prediction.labels =       {l: 0 for l in LABEL_COLS}
mock_prediction.probabilities = {l: 0.1 for l in LABEL_COLS}
mock_prediction.uncertainty =   {l: 0.02 for l in LABEL_COLS}
mock_prediction.risk_tier =     "CLEAN"
mock_prediction.flagged_labels = []
mock_prediction.requires_review = False


@pytest.fixture
def client():
    # Patch ToxicityClassifier at the class level BEFORE the app starts
    # This prevents lifespan from ever attempting to load real models
    with patch('api.model_loader.ToxicityClassifier') as MockClassifier:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = mock_prediction
        mock_instance.labels = LABEL_COLS
        MockClassifier.return_value = mock_instance

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
    assert response.status_code == 422  # Pydantic validation error


def test_predict_too_long_text_rejected(client):
    response = client.post("/predict", json={"text": "x" * 5001})
    assert response.status_code == 422