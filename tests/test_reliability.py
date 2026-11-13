import numpy as np
import pandas as pd
import pytest

from fooddemand.data import load_competition_data
from fooddemand.features import (
    MODEL_FEATURES,
    add_history_features,
    build_training_frame,
    enrich_with_auxiliary_data,
)
from fooddemand.model import build_model


def _sales_frame(dates):
    return pd.DataFrame(
        {
            "date": list(dates),
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": list(range(len(dates))),
            "onpromotion": 0,
        }
    )


def test_lags_follow_calendar_days_even_with_missing_dates():
    dates = pd.date_range("2026-01-01", periods=40)
    dates = dates[dates != pd.Timestamp("2026-01-08")]
    result = add_history_features(_sales_frame(dates)).set_index("date")

    # Jan 15 minus seven calendar days is the absent Jan 8, not Jan 7.
    assert pd.isna(result.loc["2026-01-15", "lag_7"])
    assert result.loc["2026-01-16", "lag_7"] == 7
    assert result.loc["2026-01-09", "rolling_mean_7"] == pytest.approx(3.5)


def test_duplicate_series_dates_are_rejected():
    raw = _sales_frame(pd.date_range("2026-01-01", periods=5))
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        add_history_features(raw)


def test_oil_prices_use_previous_available_calendar_day():
    sales = _sales_frame(["2026-01-05", "2026-01-06"])
    oil = pd.DataFrame({"date": ["2026-01-02", "2026-01-05"], "dcoilwtico": [70, 80]})
    result = enrich_with_auxiliary_data(sales, oil=oil)
    assert result.loc[0, "oil_price_lag_1"] == 70
    assert result.loc[1, "oil_price_lag_1"] == 80


def test_national_workday_is_not_holiday():
    sales = _sales_frame(["2026-01-05", "2026-01-06"])
    holidays = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06"],
            "locale": ["National", "National"],
            "type": ["Work Day", "Holiday"],
            "transferred": [False, False],
        }
    )
    enriched = enrich_with_auxiliary_data(sales, holidays=holidays)
    assert enriched["is_national_holiday"].tolist() == [0, 1]


def test_max_rows_keeps_complete_dates(tmp_path):
    rows = []
    for day in pd.date_range("2026-01-01", periods=3):
        for store in (1, 2):
            rows.append(
                {
                    "date": day,
                    "store_nbr": store,
                    "family": "GROCERY I",
                    "sales": 12.0,
                    "onpromotion": 0,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "train.csv", index=False)
    data = load_competition_data(tmp_path, max_rows=3)
    assert len(data.sales) == 2
    assert data.sales["date"].nunique() == 1


def test_all_missing_optional_auxiliary_columns_train(tmp_path):
    dates = pd.date_range("2026-01-01", periods=75)
    sales = _sales_frame(dates)
    sales["sales"] = 10.0 + np.sin(np.arange(len(sales)) / 3)
    frame = build_training_frame(sales)
    model = build_model()
    model.fit(frame[MODEL_FEATURES], frame["sales"])
    predictions = model.predict(frame[MODEL_FEATURES].head(2))
    assert len(predictions) == 2
    assert np.isfinite(predictions).all()


def test_negative_sales_are_rejected():
    raw = _sales_frame(pd.date_range("2026-01-01", periods=40))
    raw.loc[0, "sales"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        build_training_frame(raw)
