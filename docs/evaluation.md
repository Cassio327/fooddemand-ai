# Evaluation methodology

FoodDemand AI avoids random train/test splitting because this is a forecasting problem.

The pipeline uses rolling-origin folds. For each fold, every training date precedes the validation
window. The model is compared with two baselines:

- historical median per `(store_nbr, family)`;
- 7-day seasonal naive forecast using `lag_7`.

Metrics:

- **RMSLE**: competition-oriented and less dominated by very large sales values;
- **MAE**: average absolute error in the target unit;
- **WAPE**: aggregate absolute error relative to total actual demand.

Fast experiments with `--max-rows` retain complete calendar dates rather than cutting
arbitrary store/product series out of a day.

The current validation is rolling one-step evaluation: a row can use actual observations from
previous validation dates when forming lag features. That is useful for measuring daily updating,
but it is not the same as a fixed multi-day forecast generated all at once. The repository states this
limitation rather than presenting the score as something it is not.
