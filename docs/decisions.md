# Engineering decisions

## Gradient boosting instead of a deep model

The goal is a strong, explainable tabular forecasting baseline with fast iteration. Gradient boosting
is appropriate for mixed calendar, store and lagged numerical features and trains on CPU without a
specialized accelerator.

## Log-transformed target

The estimator is wrapped in `TransformedTargetRegressor` with `log1p/expm1`. Retail demand is skewed,
and the transformation aligns better with RMSLE while keeping the serving output in the original unit.

## Ordinal categorical encoding

Dense one-hot encoding can become memory-heavy on millions of rows. Ordinal encoding keeps the
feature matrix compact. Tree-based boosting can split on the encoded values, though this is still an
approximation; CatBoost/LightGBM are reasonable future comparisons.

## Honest metrics

CI uses synthetic data only as an integration test. Those scores are never committed as model
performance. Portfolio metrics are generated only by running the official competition dataset.
