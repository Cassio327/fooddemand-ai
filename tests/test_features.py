import pandas as pd

from fooddemand.features import add_calendar_features, add_history_features


def test_calendar_features_are_created():
    frame = pd.DataFrame({"date": ["2026-11-13"]})
    result = add_calendar_features(frame)

    assert result.loc[0, "year"] == 2026
    assert result.loc[0, "month"] == 11
    assert result.loc[0, "day"] == 13
    assert result.loc[0, "day_of_week"] == 4
    assert result.loc[0, "is_weekend"] == 0


def test_lags_use_only_previous_rows():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=30),
            "store_nbr": [1] * 30,
            "family": ["GROCERY"] * 30,
            "sales": list(range(30)),
            "onpromotion": [0] * 30,
        }
    )

    result = add_history_features(frame)

    assert pd.isna(result.loc[0, "lag_7"])
    assert result.loc[7, "lag_7"] == 0
    assert result.loc[14, "lag_14"] == 0
    assert result.loc[28, "lag_28"] == 0
    assert result.loc[7, "rolling_mean_7"] == 3
