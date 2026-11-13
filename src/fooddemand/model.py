from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from fooddemand.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_model() -> TransformedTargetRegressor:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    estimator = HistGradientBoostingRegressor(
        learning_rate=0.07,
        max_iter=220,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p, inverse_func=np.expm1)
