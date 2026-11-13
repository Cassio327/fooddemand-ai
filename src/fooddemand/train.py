from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fooddemand.backtest import rolling_origin_folds
from fooddemand.data import load_competition_data
from fooddemand.features import MODEL_FEATURES, build_training_frame
from fooddemand.metrics import regression_metrics
from fooddemand.model import build_model
from fooddemand.model_card import render_model_card

DEFAULT_MODEL = Path("models/fooddemand.joblib")
DEFAULT_REPORT = Path("reports/latest_metrics.json")
DEFAULT_MODEL_CARD = Path("reports/MODEL_CARD.md")


@dataclass
class Evaluation:
    model: object
    metrics: dict[str, dict[str, float]]


def _median_baseline(train: pd.DataFrame, validation: pd.DataFrame) -> np.ndarray:
    medians = train.groupby(["store_nbr", "family"])["sales"].median()
    global_median = float(train["sales"].median())
    return np.asarray(
        [
            float(medians.get((row.store_nbr, row.family), global_median))
            for row in validation[["store_nbr", "family"]].itertuples(index=False)
        ]
    )


def _seasonal_naive(validation: pd.DataFrame) -> np.ndarray:
    predictions = validation["lag_7"].fillna(validation["lag_28"]).fillna(0.0)
    return predictions.to_numpy(dtype=float)


def evaluate_split(train: pd.DataFrame, validation: pd.DataFrame) -> Evaluation:
    x_train = train[MODEL_FEATURES]
    y_train = train["sales"].astype(float)
    x_validation = validation[MODEL_FEATURES]
    y_validation = validation["sales"].astype(float)

    model = build_model()
    model.fit(x_train, y_train)
    model_predictions = np.clip(model.predict(x_validation), 0.0, None)

    metrics = {
        "median_baseline": regression_metrics(y_validation, _median_baseline(train, validation)),
        "seasonal_naive": regression_metrics(y_validation, _seasonal_naive(validation)),
        "model": regression_metrics(y_validation, model_predictions),
    }
    return Evaluation(model=model, metrics=metrics)


def _aggregate(folds: list[dict]) -> dict[str, dict[str, float]]:
    names = ["median_baseline", "seasonal_naive", "model"]
    metrics = ["rmsle", "mae", "wape"]
    return {
        name: {
            metric: float(np.mean([fold[name][metric] for fold in folds]))
            for metric in metrics
        }
        for name in names
    }


def _log_mlflow(report: dict, model_path: Path) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("Install the tracking extra: pip install -e '.[tracking]'") from exc

    with mlflow.start_run(run_name="fooddemand-backtest"):
        mlflow.log_params(
            {
                "backtest_folds": report["backtest"]["folds"],
                "validation_days": report["backtest"]["validation_days"],
                "rows_model_ready": report["rows"]["model_ready"],
            }
        )
        for model_name, values in report["backtest"]["aggregate"].items():
            for metric_name, value in values.items():
                mlflow.log_metric(f"{model_name}_{metric_name}", value)
        mlflow.log_artifact(str(model_path))


def train_project(
    data_dir: str | Path,
    max_rows: int | None = None,
    backtest_folds: int = 3,
    validation_days: int = 16,
    data_source: str = "unverified",
) -> tuple[dict, dict]:
    if data_source not in {"unverified", "kaggle-store-sales", "synthetic"}:
        raise ValueError("Invalid data source")
    data = load_competition_data(data_dir, max_rows=max_rows)
    frame = build_training_frame(
        data.sales, data.stores, data.oil, data.holidays, data.transactions
    )
    folds = rolling_origin_folds(frame, folds=backtest_folds, validation_days=validation_days)

    fold_metrics: list[dict] = []
    for fold in folds:
        evaluation = evaluate_split(fold.train, fold.validation)
        fold_metrics.append(
            {
                "fold": fold.index,
                "train_end": fold.train["date"].max().date().isoformat(),
                "validation_start": fold.validation["date"].min().date().isoformat(),
                "validation_end": fold.validation["date"].max().date().isoformat(),
                **evaluation.metrics,
            }
        )

    final_cutoff = frame["date"].max() - pd.Timedelta(days=validation_days - 1)
    final_train = frame[frame["date"] < final_cutoff].copy()
    final_validation = frame[frame["date"] >= final_cutoff].copy()
    holdout = evaluate_split(final_train, final_validation)

    # Refit on every model-ready row for the serving artifact.
    final_model = build_model()
    final_model.fit(frame[MODEL_FEATURES], frame["sales"].astype(float))

    aggregate = _aggregate(fold_metrics)
    baseline = aggregate["seasonal_naive"]
    model_metrics = aggregate["model"]
    improvement = {
        key: (
            float((baseline[key] - model_metrics[key]) / baseline[key])
            if baseline[key] not in (0.0, float("inf"))
            else 0.0
        )
        for key in model_metrics
    }

    report = {
        "dataset": (
            "Kaggle Store Sales - Time Series Forecasting"
            if data_source == "kaggle-store-sales"
            else "Synthetic integration fixture" if data_source == "synthetic"
            else "Unverified user-provided dataset"
        ),
        "data_source": data_source,
        "evaluation": "rolling one-step chronological backtesting",
        "rows": {"raw": int(len(data.sales)), "model_ready": int(len(frame))},
        "date_range": {
            "start": frame["date"].min().date().isoformat(),
            "end": frame["date"].max().date().isoformat(),
        },
        "features": MODEL_FEATURES,
        "backtest": {
            "folds": backtest_folds,
            "validation_days": validation_days,
            "results": fold_metrics,
            "aggregate": aggregate,
            "improvement_vs_seasonal_naive": improvement,
        },
        "holdout": holdout.metrics,
        "limitations": [
            (
                "Backtesting is rolling one-step evaluation and can use actual earlier "
                "validation observations in lag features."
            ),
            "A production multi-step forecast requires recursive or direct horizon forecasting.",
            "The public Ecuadorian grocery dataset is a portfolio proxy, not iFood data.",
        ],
    }
    artifact = {
        "model": final_model,
        "metadata": {
            "version": "1.0.1",
            "data_source": data_source,
            "features": MODEL_FEATURES,
            "evaluation": report["evaluation"],
            "backtest_aggregate": aggregate,
            "trained_through": report["date_range"]["end"],
        },
    }
    return artifact, report


def save_outputs(
    artifact: dict,
    report: dict,
    model_path: Path,
    report_path: Path,
    model_card_path: Path,
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    model_card_path.write_text(render_model_card(report), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and backtest FoodDemand AI.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--model-card-out", type=Path, default=DEFAULT_MODEL_CARD)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--backtest-folds", type=int, default=3)
    parser.add_argument("--validation-days", type=int, default=16)
    parser.add_argument(
        "--data-source",
        choices=["unverified", "synthetic", "kaggle-store-sales"],
        default="unverified",
        help="Explicitly label the input dataset; do not publish synthetic metrics as real.",
    )
    parser.add_argument("--mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact, report = train_project(
        data_dir=args.data_dir,
        max_rows=args.max_rows,
        backtest_folds=args.backtest_folds,
        validation_days=args.validation_days,
        data_source=args.data_source,
    )
    save_outputs(artifact, report, args.model_out, args.report_out, args.model_card_out)
    if args.mlflow:
        _log_mlflow(report, args.model_out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
