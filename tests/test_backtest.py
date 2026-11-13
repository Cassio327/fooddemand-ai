import pandas as pd

from fooddemand.backtest import rolling_origin_folds


def test_backtest_windows_are_chronological():
    frame = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=100), "sales": range(100)})
    folds = rolling_origin_folds(frame, folds=3, validation_days=10)
    assert len(folds) == 3
    for fold in folds:
        assert fold.train["date"].max() < fold.validation["date"].min()
