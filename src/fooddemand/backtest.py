from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    index: int
    train: pd.DataFrame
    validation: pd.DataFrame


def rolling_origin_folds(
    frame: pd.DataFrame,
    folds: int = 3,
    validation_days: int = 16,
) -> list[Fold]:
    if folds < 1:
        raise ValueError("folds must be at least 1")
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")

    dated = frame.copy()
    dated["date"] = pd.to_datetime(dated["date"], errors="raise")
    last_date = dated["date"].max().normalize()
    output: list[Fold] = []

    for reverse_index in range(folds - 1, -1, -1):
        validation_end = last_date - pd.Timedelta(days=reverse_index * validation_days)
        validation_start = validation_end - pd.Timedelta(days=validation_days - 1)
        train = dated[dated["date"] < validation_start].copy()
        validation = dated[
            (dated["date"] >= validation_start) & (dated["date"] <= validation_end)
        ].copy()
        if train.empty or validation.empty:
            raise ValueError("Not enough history to create all requested backtest folds")
        output.append(Fold(index=len(output) + 1, train=train, validation=validation))
    return output
