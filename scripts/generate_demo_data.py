from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(output_dir: Path, days: int = 120) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    for store in (1, 2, 3):
        for family, base in (("GROCERY I", 80.0), ("BEVERAGES", 55.0), ("PRODUCE", 65.0)):
            for index, current_date in enumerate(pd.date_range("2026-01-01", periods=days)):
                promo = int(index % 9 == 0) * int(rng.integers(1, 5))
                weekend = 12.0 if current_date.dayofweek >= 5 else 0.0
                trend = 0.05 * index
                sales = max(0.0, base + weekend + promo * 4 + trend + rng.normal(0, 3))
                rows.append(
                    {
                        "id": len(rows),
                        "date": current_date.date().isoformat(),
                        "store_nbr": store,
                        "family": family,
                        "sales": sales,
                        "onpromotion": promo,
                    }
                )
    pd.DataFrame(rows).to_csv(output_dir / "train.csv", index=False)

    pd.DataFrame(
        {
            "store_nbr": [1, 2, 3],
            "city": ["Quito", "Guayaquil", "Cuenca"],
            "state": ["Pichincha", "Guayas", "Azuay"],
            "type": ["A", "B", "C"],
            "cluster": [1, 2, 3],
        }
    ).to_csv(output_dir / "stores.csv", index=False)

    dates = pd.date_range("2026-01-01", periods=days)
    pd.DataFrame(
        {"date": dates.date, "dcoilwtico": 70 + np.sin(np.arange(days) / 8) * 4}
    ).to_csv(output_dir / "oil.csv", index=False)

    transactions = []
    for store in (1, 2, 3):
        for index, current_date in enumerate(dates):
            transactions.append(
                {"date": current_date.date(), "store_nbr": store, "transactions": 500 + store * 20 + index}
            )
    pd.DataFrame(transactions).to_csv(output_dir / "transactions.csv", index=False)

    pd.DataFrame(
        [
            {
                "date": "2026-02-16",
                "type": "Holiday",
                "locale": "National",
                "locale_name": "Ecuador",
                "description": "Synthetic holiday",
                "transferred": False,
            }
        ]
    ).to_csv(output_dir / "holidays_events.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--days", type=int, default=120)
    args = parser.parse_args()
    generate(args.output_dir, args.days)
