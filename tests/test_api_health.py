from fastapi.testclient import TestClient

from vectramark.api.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    payload = response.json()

    assert payload["application"] == "VectraMark"
    assert payload["status"] == "running"


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert payload["application"] == "VectraMark"
    assert "timestamp" in payload


def test_openapi_documentation_is_available():
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "paths" in response.json()
