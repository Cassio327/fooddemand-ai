import math

from fooddemand.metrics import regression_metrics, rmsle, wape


def test_perfect_predictions_have_zero_error():
    metrics = regression_metrics([0, 1, 10], [0, 1, 10])
    assert metrics == {"rmsle": 0.0, "mae": 0.0, "wape": 0.0}


def test_negative_predictions_are_clipped_for_rmsle():
    assert rmsle([0, 1], [-5, 1]) == 0.0


def test_wape_handles_zero_denominator():
    assert wape([0, 0], [0, 0]) == 0.0
    assert math.isinf(wape([0, 0], [1, 0]))
