from pathlib import Path

from fooddemand.train import train_project
from scripts.generate_demo_data import generate


def test_complete_training_pipeline(tmp_path: Path):
    data_dir = tmp_path / "data"
    generate(data_dir, days=90)
    artifact, report = train_project(data_dir, backtest_folds=2, validation_days=10)

    assert "model" in artifact
    assert report["rows"]["model_ready"] > 0
    assert len(report["backtest"]["results"]) == 2
    assert set(report["backtest"]["aggregate"]["model"]) == {"rmsle", "mae", "wape"}
