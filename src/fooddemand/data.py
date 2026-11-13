from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class CompetitionData:
    sales: pd.DataFrame
    stores: pd.DataFrame | None = None
    oil: pd.DataFrame | None = None
    holidays: pd.DataFrame | None = None
    transactions: pd.DataFrame | None = None


def _read_optional(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def load_competition_data(data_dir: str | Path, max_rows: int | None = None) -> CompetitionData:
    root = Path(data_dir)
    train_path = root / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}. Run scripts/download_data.sh first.")

    sales = pd.read_csv(train_path)
    sales["date"] = pd.to_datetime(sales["date"], errors="raise")
    if max_rows is not None and len(sales) > max_rows:
        sales = sales.sort_values("date").tail(max_rows).copy()

    return CompetitionData(
        sales=sales,
        stores=_read_optional(root / "stores.csv"),
        oil=_read_optional(root / "oil.csv"),
        holidays=_read_optional(root / "holidays_events.csv"),
        transactions=_read_optional(root / "transactions.csv"),
    )
