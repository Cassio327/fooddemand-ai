# Engineering decisions

## Gradient boosting instead of a deep model

The goal is a strong, explainable tabular forecasting baseline with fast iteration. Gradient boosting
is appropriate for mixed calendar, store and lagged numerical features and trains on CPU without a
specialized accelerator.

## Log-transformed target

The estimator is wrapped in `TransformedTargetRegressor` with `log1p/expm1`. Retail demand is skewed,
and the transformation aligns better with RMSLE while keeping the serving output in the original unit.

## Native categorical splits

Dense one-hot encoding can become memory-heavy on millions of rows. Ordinal encoding keeps the
feature matrix compact, but categorical codes have no intrinsic numerical meaning. The estimator
now enables **native categorical splits** for the encoded categorical columns. CatBoost and
LightGBM remain useful future benchmark candidates.

## Honest metrics

CI uses synthetic data only as an integration test. Those scores are never committed as model
performance. Portfolio metrics are generated only by running the official competition dataset.
