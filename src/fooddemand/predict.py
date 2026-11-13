from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from fooddemand.features import MODEL_FEATURES, add_calendar_features


def load_model(path: str | Path):
    return joblib.load(path)


def build_inference_frame(payload: dict) -> pd.DataFrame:
    frame = pd.DataFrame([payload])
    frame = add_calendar_features(frame)

    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing inference features: {', '.join(missing)}")

    return frame[MODEL_FEATURES]


def predict_one(model, payload: dict) -> float:
    frame = build_inference_frame(payload)
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)
