from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fooddemand.predict import load_artifact, predict_many, predict_one

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fooddemand.joblib"))

app = FastAPI(
    title="FoodDemand AI",
    version="1.0.0",
    description="Retail demand forecasting API with a versioned ML artifact.",
)


class ForecastRequest(BaseModel):
    date: date
    store_nbr: int = Field(ge=1)
    family: str = Field(min_length=1)
    onpromotion: int = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_14: float = Field(ge=0)
    lag_28: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    store_type: str = "UNKNOWN"
    city: str = "UNKNOWN"
    state: str = "UNKNOWN"
    cluster: int = -1
    oil_price_lag_1: float | None = None
    transactions_lag_1: float | None = None
    is_national_holiday: int = Field(default=0, ge=0, le=1)


class ForecastResponse(BaseModel):
    predicted_sales: float
    model_version: str


@lru_cache(maxsize=1)
def get_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run the training command first."
        )
    return load_artifact(MODEL_PATH)


def _payload(request: ForecastRequest) -> dict:
    values = request.model_dump()
    values["date"] = request.date.isoformat()
    return values


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ready": MODEL_PATH.exists(), "model_path": str(MODEL_PATH)}


@app.get("/model")
def model_info() -> dict:
    try:
        return get_artifact().get("metadata", {})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    try:
        artifact = get_artifact()
        prediction = predict_one(artifact, _payload(request))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    version = str(artifact.get("metadata", {}).get("version", "unknown"))
    return ForecastResponse(predicted_sales=prediction, model_version=version)


@app.post("/predict/batch")
def predict_batch(requests: list[ForecastRequest]) -> dict:
    if not requests:
        raise HTTPException(status_code=422, detail="At least one request is required")
    if len(requests) > 100:
        raise HTTPException(status_code=413, detail="Batch size is limited to 100")
    try:
        artifact = get_artifact()
        predictions = predict_many(artifact, [_payload(request) for request in requests])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "predictions": predictions,
        "model_version": artifact.get("metadata", {}).get("version", "unknown"),
    }
