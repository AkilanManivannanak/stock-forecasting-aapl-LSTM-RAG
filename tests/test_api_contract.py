import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
import app as app_module


class FakeScaler:
    def transform(self, x):
        return np.array(x, dtype=float)

    def inverse_transform(self, x):
        return np.array(x, dtype=float)


def _fake_df(n=200):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n),
            "Volume": np.linspace(1_000_000, 1_100_000, n),
        },
        index=idx,
    )


def test_health():
    client = TestClient(app_module.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_forecast_contract(monkeypatch):
    client = TestClient(app_module.app)

    monkeypatch.setattr(app_module, "get_latest_data", lambda ticker: _fake_df())
    monkeypatch.setattr(app_module, "load_trained_model", lambda ticker: object())
    monkeypatch.setattr(app_module, "load_scaler", lambda ticker: FakeScaler())
    monkeypatch.setattr(
        app_module,
        "build_latest_window_scaled",
        lambda df, scaler: np.zeros((1, app_module.TIME_STEP, 1), dtype=float),
    )
    monkeypatch.setattr(app_module, "make_forecast", lambda model, X_last, horizon: float(horizon))

    r = client.post("/forecast", json={"ticker": "AAPL", "days": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["days"] == 5
    assert len(body["points"]) == 5
    assert all("date" in p and "forecast" in p for p in body["points"])


def test_multi_horizon_contract(monkeypatch):
    client = TestClient(app_module.app)

    monkeypatch.setattr(app_module, "get_latest_data", lambda ticker: _fake_df())
    monkeypatch.setattr(app_module, "load_trained_model", lambda ticker: object())
    monkeypatch.setattr(app_module, "load_scaler", lambda ticker: FakeScaler())
    monkeypatch.setattr(
        app_module,
        "build_latest_window_scaled",
        lambda df, scaler: np.zeros((1, app_module.TIME_STEP, 1), dtype=float),
    )
    monkeypatch.setattr(app_module, "make_forecast", lambda model, X_last, horizon: float(horizon))

    r = client.post("/multi_horizon_forecast", json={"ticker": "AAPL", "days": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["requested_days"] == 10
    assert len(body["horizons"]) == 10


def test_ask_contract(monkeypatch):
    client = TestClient(app_module.app)

    def fake_answer_question(**kwargs):
        return {
            "answer": "Test answer",
            "citations": [{"source": "data/kb/PROJECT_GOAL.md", "chunk_id": "x", "page": None, "snippet": "y"}],
            "retrieval_debug": {"docs_returned": 1},
        }

    monkeypatch.setattr(app_module, "answer_question", lambda **kwargs: fake_answer_question(**kwargs))

    r = client.post("/ask", json={"ticker": "AAPL", "question": "What is the goal?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "citations" in body
    assert "retrieval_debug" in body
