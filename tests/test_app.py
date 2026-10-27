"""
Unit tests for the Iris Flower Classifier web application.
"""

import json
import pytest
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home_page(client):
    """Test that the home page loads successfully."""
    response = client.get("/")
    assert response.status_code == 200


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "version" in data


def test_predict_valid_input(client):
    """Test prediction with valid Iris measurements."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "species" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["species"] in ["setosa", "versicolor", "virginica"]


def test_predict_setosa(client):
    """Test prediction for Iris Setosa (small petals)."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.0,
            "sepal_width": 3.4,
            "petal_length": 1.5,
            "petal_width": 0.2,
        },
    )
    data = json.loads(response.data)
    assert data["species"] == "setosa"


def test_predict_versicolor(client):
    """Test prediction for Iris Versicolor (medium petals)."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.9,
            "sepal_width": 2.8,
            "petal_length": 4.1,
            "petal_width": 1.3,
        },
    )
    data = json.loads(response.data)
    assert data["species"] == "versicolor"


def test_predict_virginica(client):
    """Test prediction for Iris Virginica (large petals)."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": 6.7,
            "sepal_width": 3.1,
            "petal_length": 5.6,
            "petal_width": 2.4,
        },
    )
    data = json.loads(response.data)
    assert data["species"] == "virginica"


def test_predict_invalid_input(client):
    """Test prediction with missing fields."""
    response = client.post(
        "/predict",
        json={"sepal_length": 5.1},
    )
    assert response.status_code == 400


def test_predict_non_numeric(client):
    """Test prediction with non-numeric input."""
    response = client.post(
        "/predict",
        json={
            "sepal_length": "abc",
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )
    assert response.status_code == 400
