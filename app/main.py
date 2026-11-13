from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fooddemand.predict import load_model, predict_one

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/fooddemand.joblib"))

app = FastAPI(
    title="FoodDemand AI",
    version="0.1.0",
    description="Retail demand forecasting API.",
)


class ForecastRequest(BaseModel):
    date: str = Field(examples=["2017-08-16"])
    store_nbr: int = Field(ge=1)
    family: str = Field(min_length=1)
    onpromotion: int = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_14: float = Field(ge=0)
    lag_28: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)


class ForecastResponse(BaseModel):
    predicted_sales: float
    model_version: str = "0.1.0"


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run the training command first."
        )
    return load_model(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(request: ForecastRequest) -> ForecastResponse:
    try:
        model = get_model()
        prediction = predict_one(model, request.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ForecastResponse(predicted_sales=prediction)
