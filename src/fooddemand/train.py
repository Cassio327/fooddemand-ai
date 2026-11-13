from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fooddemand.features import MODEL_FEATURES, build_training_frame
from fooddemand.metrics import regression_metrics
from fooddemand.model import build_model

DEFAULT_DATA = Path("data/raw/train.csv")
DEFAULT_MODEL = Path("models/fooddemand.joblib")
DEFAULT_REPORT = Path("reports/latest_metrics.json")


@dataclass
class TrainingResult:
    model: object
    report: dict


def load_training_data(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if max_rows is not None and len(frame) > max_rows:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.sort_values("date").tail(max_rows).copy()
    return frame


def temporal_split(
    frame: pd.DataFrame,
    validation_days: int = 28,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_days < 1:
        raise ValueError("validation_days must be at least 1")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    last_date = result["date"].max()
    cutoff = last_date - pd.Timedelta(days=validation_days - 1)

    train = result[result["date"] < cutoff].copy()
    validation = result[result["date"] >= cutoff].copy()

    if train.empty or validation.empty:
        raise ValueError("Temporal split produced an empty train or validation set")

    return train, validation


def _median_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    medians = train.groupby(["store_nbr", "family"])["sales"].median()
    global_median = float(train["sales"].median())

    predictions = []
    for row in validation[["store_nbr", "family"]].itertuples(index=False):
        predictions.append(float(medians.get((row.store_nbr, row.family), global_median)))
    return np.asarray(predictions)


def train_model(raw_frame: pd.DataFrame, validation_days: int = 28) -> TrainingResult:
    frame = build_training_frame(raw_frame)
    train, validation = temporal_split(frame, validation_days=validation_days)

    x_train = train[MODEL_FEATURES]
    y_train = train["sales"].astype(float)
    x_validation = validation[MODEL_FEATURES]
    y_validation = validation["sales"].astype(float)

    baseline_predictions = _median_baseline(train, validation)

    model = build_model()
    model.fit(x_train, y_train)
    predictions = np.clip(model.predict(x_validation), a_min=0.0, a_max=None)

    baseline_metrics = regression_metrics(y_validation, baseline_predictions)
    model_metrics = regression_metrics(y_validation, predictions)

    report = {
        "evaluation": "rolling one-step temporal holdout",
        "validation_days": validation_days,
        "rows": {
            "raw": int(len(raw_frame)),
            "model_ready": int(len(frame)),
            "train": int(len(train)),
            "validation": int(len(validation)),
        },
        "dates": {
            "train_start": train["date"].min().date().isoformat(),
            "train_end": train["date"].max().date().isoformat(),
            "validation_start": validation["date"].min().date().isoformat(),
            "validation_end": validation["date"].max().date().isoformat(),
        },
        "baseline": baseline_metrics,
        "model": model_metrics,
        "improvement_vs_baseline": {
            metric: (
                float((baseline_metrics[metric] - model_metrics[metric]) / baseline_metrics[metric])
                if baseline_metrics[metric] not in (0.0, float("inf"))
                else 0.0
            )
            for metric in model_metrics
        },
        "features": MODEL_FEATURES,
    }
    return TrainingResult(model=model, report=report)


def save_artifacts(
    result: TrainingResult,
    model_path: Path = DEFAULT_MODEL,
    report_path: Path = DEFAULT_REPORT,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, model_path)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the FoodDemand AI forecasting model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validation-days", type=int, default=28)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the most recent N rows for a faster local experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_frame = load_training_data(args.data, max_rows=args.max_rows)
    result = train_model(raw_frame, validation_days=args.validation_days)
    save_artifacts(result, args.model_out, args.report_out)
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
