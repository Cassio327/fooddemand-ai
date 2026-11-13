import numpy as np
import pandas as pd

from fooddemand.train import train_model


def make_synthetic_frame(days: int = 75) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)

    for store_nbr in (1, 2):
        for family, base in (("GROCERY", 30.0), ("BEVERAGES", 45.0)):
            for index, date in enumerate(pd.date_range("2026-01-01", periods=days)):
                weekly = 5.0 if date.dayofweek >= 5 else 0.0
                promotion = int(index % 10 == 0)
                sales = base + weekly + promotion * 8 + rng.normal(0, 1)
                rows.append(
                    {
                        "date": date,
                        "store_nbr": store_nbr,
                        "family": family,
                        "sales": max(0.0, sales),
                        "onpromotion": promotion,
                    }
                )

    return pd.DataFrame(rows)


def test_training_pipeline_returns_metrics():
    result = train_model(make_synthetic_frame(), validation_days=14)

    assert set(result.report["model"]) == {"rmsle", "mae", "wape"}
    assert result.report["rows"]["train"] > 0
    assert result.report["rows"]["validation"] > 0
    assert result.report["model"]["mae"] >= 0
