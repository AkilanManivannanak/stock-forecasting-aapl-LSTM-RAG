from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_forecast_basic():
    r = client.post("/forecast", json={"ticker": "AAPL", "days": 5})
    assert r.status_code == 200
    data = r.json()
    assert "points" in data
    assert len(data["points"]) == 5
