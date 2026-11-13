# Data

This project uses the **Store Sales - Time Series Forecasting** Kaggle competition dataset.

The raw competition files are intentionally not committed to Git. After accepting the competition
rules and configuring your Kaggle credentials, run:

```bash
pip install -e ".[data]"
bash scripts/download_data.sh
```

The training pipeline currently requires `data/raw/train.csv` with these columns:

- `date`
- `store_nbr`
- `family`
- `sales`
- `onpromotion`

Additional competition tables such as oil, holidays, transactions and store metadata are planned
for later feature-engineering iterations.
