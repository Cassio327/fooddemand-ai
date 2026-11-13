from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fooddemand.features import MODEL_FEATURES, add_calendar_features


def load_artifact(path: str | Path) -> dict:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        return {"model": artifact, "metadata": {"legacy_artifact": True}}
    return artifact


def build_inference_frame(payloads: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(payloads)
    frame = add_calendar_features(frame)

    defaults: dict[str, object] = {
        "store_type": "UNKNOWN",
        "city": "UNKNOWN",
        "state": "UNKNOWN",
        "cluster": -1,
        "oil_price_lag_1": float("nan"),
        "transactions_lag_1": float("nan"),
        "is_national_holiday": 0,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {', '.join(missing)}")
    return frame[MODEL_FEATURES]


def predict_many(artifact: dict, payloads: list[dict]) -> list[float]:
    frame = build_inference_frame(payloads)
    predictions = artifact["model"].predict(frame)
    return [max(0.0, float(value)) for value in predictions]


def predict_one(artifact: dict, payload: dict) -> float:
    return predict_many(artifact, [payload])[0]
