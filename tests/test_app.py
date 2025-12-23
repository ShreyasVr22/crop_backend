from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_import_app():
    # Ensure the FastAPI app object imports correctly
    assert app is not None


def test_health_endpoint():
    # Ensure the /health endpoint responds with status OK
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "ok"
    assert "districts_loaded" in data
