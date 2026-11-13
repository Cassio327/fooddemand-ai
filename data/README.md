# Data

FoodDemand AI uses the **Store Sales - Time Series Forecasting** Kaggle competition dataset.
Raw competition files are intentionally excluded from Git.

## Download

1. Sign in to Kaggle and accept the competition rules.
2. Configure the Kaggle API credentials.
3. Run:

```bash
pip install -e ".[data]"
bash scripts/download_data.sh
```

Expected files include `train.csv`, `stores.csv`, `oil.csv`, `transactions.csv` and
`holidays_events.csv` under `data/raw/`.

For tests and a no-credential demo, `scripts/generate_demo_data.py` creates a fully synthetic dataset.
Synthetic results must never be presented as competition performance.
