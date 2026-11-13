from fooddemand.model_card import render_model_card


def test_model_card_renders_metrics():
    report = {
        "backtest": {
            "aggregate": {
                "seasonal_naive": {"rmsle": 0.5, "mae": 2.0, "wape": 0.2},
                "model": {"rmsle": 0.4, "mae": 1.5, "wape": 0.15},
            }
        }
    }
    card = render_model_card(report)
    assert "0.4000" in card
    assert "Seasonal naive" in card
