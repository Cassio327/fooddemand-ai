from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_log_error


def _as_non_negative(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), a_min=0.0, a_max=None)


def rmsle(y_true, y_pred) -> float:
    truth = _as_non_negative(y_true)
    prediction = _as_non_negative(y_pred)
    return float(np.sqrt(mean_squared_log_error(truth, prediction)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


def wape(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)
    denominator = float(np.abs(truth).sum())
    if denominator == 0.0:
        return 0.0 if float(np.abs(truth - prediction).sum()) == 0.0 else float("inf")
    return float(np.abs(truth - prediction).sum() / denominator)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {"rmsle": rmsle(y_true, y_pred), "mae": mae(y_true, y_pred), "wape": wape(y_true, y_pred)}
