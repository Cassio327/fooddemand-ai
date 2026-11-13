from datetime import date

from fastapi.testclient import TestClient

import app.main as api
from fooddemand.predict import build_inference_frame


class ConstantModel:
    def predict(self, frame):
        return [42.0] * len(frame)


def test_health_works_without_model(tmp_path, monkeypatch):
    missing = tmp_path / "missing.joblib"
    monkeypatch.setattr(api, "MODEL_PATH", missing)
    api.get_artifact.cache_clear()
    response = TestClient(api.app).get("/health")
    assert response.status_code == 200
    assert response.json()["model_file_present"] is False


def test_predict_with_cached_artifact(monkeypatch):
    artifact = {"model": ConstantModel(), "metadata": {"version": "test"}}
    monkeypatch.setattr(api, "get_artifact", lambda: artifact)
    payload = {
        "date": date(2026, 11, 13).isoformat(),
        "store_nbr": 1,
        "family": "GROCERY I",
        "onpromotion": 0,
        "lag_7": 40,
        "lag_14": 41,
        "lag_28": 39,
        "rolling_mean_7": 40,
        "rolling_mean_28": 39,
    }
    response = TestClient(api.app).post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json() == {"predicted_sales": 42.0, "model_version": "test"}


def test_optional_numeric_fields_accept_none():
    frame = build_inference_frame(
        [
            {
                "date": "2026-11-13",
                "store_nbr": 1,
                "family": "GROCERY I",
                "onpromotion": 0,
                "lag_7": 40,
                "lag_14": 41,
                "lag_28": 39,
                "rolling_mean_7": 40,
                "rolling_mean_28": 39,
                "oil_price_lag_1": None,
                "transactions_lag_1": None,
            }
        ]
    )
    assert frame["oil_price_lag_1"].isna().all()
    assert frame["transactions_lag_1"].isna().all()


def test_readiness_returns_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "absent.joblib")
    api.get_artifact.cache_clear()
    response = TestClient(api.app).get("/ready")
    assert response.status_code == 503


def test_non_finite_lag_is_rejected():
    payload = {
        "date": "2017-08-16",
        "store_nbr": 1,
        "family": "GROCERY I",
        "onpromotion": 0,
        "lag_7": "NaN",
        "lag_14": 40,
        "lag_28": 40,
        "rolling_mean_7": 40,
        "rolling_mean_28": 40,
    }
    assert TestClient(api.app).post("/predict", json=payload).status_code == 422
