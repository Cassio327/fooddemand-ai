from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

GROUP_COLUMNS = ["store_nbr", "family"]
LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)

MODEL_FEATURES = [
    "store_nbr",
    "family",
    "onpromotion",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
]


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic calendar features from the date column."""
    _require_columns(frame, ["date"])
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day"] = result["date"].dt.day.astype("int8")
    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    result["day_of_year"] = result["date"].dt.dayofyear.astype("int16")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    return result


def add_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe historical features using only earlier observations per series."""
    _require_columns(frame, [*GROUP_COLUMNS, "date", "sales"])
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values([*GROUP_COLUMNS, "date"]).reset_index(drop=True)

    grouped = result.groupby(GROUP_COLUMNS, sort=False)["sales"]

    for lag in LAG_DAYS:
        result[f"lag_{lag}"] = grouped.shift(lag)

    for window in ROLLING_WINDOWS:
        result[f"rolling_mean_{window}"] = grouped.transform(
            lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
        )

    return result


def build_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready rows and drop dates without enough history for the longest lag."""
    _require_columns(frame, ["date", "store_nbr", "family", "sales", "onpromotion"])
    result = add_history_features(frame)
    result = add_calendar_features(result)
    result = result.dropna(subset=[f"lag_{max(LAG_DAYS)}"]).reset_index(drop=True)
    return result
