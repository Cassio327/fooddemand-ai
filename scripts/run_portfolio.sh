#!/usr/bin/env bash
set -euo pipefail

python -m fooddemand.eda --data-dir data/raw
python -m fooddemand.train --data-dir data/raw --backtest-folds 3 --validation-days 16 "$@"

echo "Portfolio run complete. Review reports/latest_metrics.json and reports/MODEL_CARD.md."
