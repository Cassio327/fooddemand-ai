#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found."
  echo "Install project data extras with: pip install -e '.[data]'"
  exit 1
fi

echo "Downloading Store Sales - Time Series Forecasting..."
kaggle competitions download -c store-sales-time-series-forecasting -p data/raw

unzip -o data/raw/store-sales-time-series-forecasting.zip -d data/raw
rm data/raw/store-sales-time-series-forecasting.zip

echo "Dataset available under data/raw/."
