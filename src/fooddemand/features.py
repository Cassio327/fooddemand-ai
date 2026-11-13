from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

GROUP_COLUMNS = ["store_nbr", "family"]
LAG_DAYS = (7, 14, 28)
ROLLING_WINDOWS = (7, 28)

CATEGORICAL_FEATURES = ["family", "store_type", "city", "state"]
NUMERIC_FEATURES = [
    "store_nbr",
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
    "cluster",
    "oil_price_lag_1",
    "transactions_lag_1",
    "is_national_holiday",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    """Compute history by calendar day rather than by position in the source file.

    A day missing from the input is not silently treated as the previous day.
    Forecast inputs never include sales observed on the forecast date itself.
    """
    _require_columns(frame, [*GROUP_COLUMNS, "date", "sales"])
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values([*GROUP_COLUMNS, "date"]).reset_index(drop=True)

    if result.duplicated([*GROUP_COLUMNS, "date"]).any():
        raise ValueError("Duplicate store/family/date observations are not supported")

    feature_names = [
        *(f"lag_{lag}" for lag in LAG_DAYS),
        *(f"rolling_mean_{window}" for window in ROLLING_WINDOWS),
    ]
    for name in feature_names:
        result[name] = np.nan

    for _, series_rows in result.groupby(GROUP_COLUMNS, sort=False):
        calendar = series_rows.set_index("date")["sales"].asfreq("D")
        dates = series_rows["date"]
        positions = series_rows.index

        for lag in LAG_DAYS:
            values = calendar.shift(lag).reindex(dates).to_numpy(dtype=float)
            result.loc[positions, f"lag_{lag}"] = values

        historical = calendar.shift(1)
        for window in ROLLING_WINDOWS:
            values = (
                historical.rolling(window=window, min_periods=1)
                .mean()
                .reindex(dates)
                .to_numpy(dtype=float)
            )
            result.loc[positions, f"rolling_mean_{window}"] = values

    return result


def enrich_with_auxiliary_data(
    sales: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    oil: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge competition metadata while avoiding same-day target proxies where possible."""
    result = sales.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")

    if stores is not None and not stores.empty:
        store_frame = stores.copy().rename(columns={"type": "store_type"})
        wanted = ["store_nbr", "city", "state", "store_type", "cluster"]
        _require_columns(store_frame, wanted)
        result = result.merge(
            store_frame[wanted], on="store_nbr", how="left", validate="many_to_one"
        )

    if oil is not None and not oil.empty:
        oil_frame = oil.copy()
        _require_columns(oil_frame, ["date", "dcoilwtico"])
        oil_frame["date"] = pd.to_datetime(oil_frame["date"], errors="raise")
        if oil_frame["date"].duplicated().any():
            raise ValueError("Duplicate oil dates are not supported")
        oil_values = oil_frame.set_index("date")["dcoilwtico"].sort_index()
        oil_values = pd.to_numeric(oil_values, errors="coerce")
        all_days = pd.date_range(
            min(result["date"].min(), oil_values.index.min()),
            result["date"].max(),
            freq="D",
        )
        # A quote on day D becomes available to the model on day D+1.
        # Forward-fill closures/weekends using only previously observed quotes.
        previous_day_quotes = oil_values.reindex(all_days).ffill().shift(1)
        result["oil_price_lag_1"] = result["date"].map(previous_day_quotes)

    if transactions is not None and not transactions.empty:
        tx = transactions.copy()
        _require_columns(tx, ["date", "store_nbr", "transactions"])
        tx["date"] = pd.to_datetime(tx["date"], errors="raise") + pd.Timedelta(days=1)
        tx = tx.rename(columns={"transactions": "transactions_lag_1"})
        result = result.merge(
            tx[["date", "store_nbr", "transactions_lag_1"]],
            on=["date", "store_nbr"],
            how="left",
            validate="many_to_one",
        )

    if holidays is not None and not holidays.empty:
        holiday_frame = holidays.copy()
        _require_columns(holiday_frame, ["date", "locale", "transferred", "type"])
        holiday_frame["date"] = pd.to_datetime(holiday_frame["date"], errors="raise")
        transferred = holiday_frame["transferred"].astype(str).str.lower().eq("true")
        national = holiday_frame[
            holiday_frame["locale"].eq("National")
            & ~transferred
            & ~holiday_frame["type"].eq("Work Day")
        ]
        holiday_dates = national[["date"]].drop_duplicates()
        holiday_dates["is_national_holiday"] = 1
        result = result.merge(holiday_dates, on="date", how="left", validate="many_to_one")

    defaults: dict[str, object] = {
        "store_type": "UNKNOWN",
        "city": "UNKNOWN",
        "state": "UNKNOWN",
        "cluster": -1,
        "oil_price_lag_1": np.nan,
        "transactions_lag_1": np.nan,
        "is_national_holiday": 0,
    }
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
        elif isinstance(default, str):
            result[column] = result[column].fillna(default)
        elif column == "is_national_holiday":
            result[column] = result[column].fillna(default)

    result["is_national_holiday"] = result["is_national_holiday"].astype("int8")
    return result


def build_training_frame(
    sales: pd.DataFrame,
    stores: pd.DataFrame | None = None,
    oil: pd.DataFrame | None = None,
    holidays: pd.DataFrame | None = None,
    transactions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    _require_columns(sales, ["date", "store_nbr", "family", "sales", "onpromotion"])
    if sales["sales"].isna().any() or (sales["sales"] < 0).any():
        raise ValueError("Sales targets must be present and non-negative")
    result = enrich_with_auxiliary_data(sales, stores, oil, holidays, transactions)
    result = add_history_features(result)
    result = add_calendar_features(result)
    result = result.dropna(subset=[f"lag_{max(LAG_DAYS)}"]).reset_index(drop=True)
    return result
