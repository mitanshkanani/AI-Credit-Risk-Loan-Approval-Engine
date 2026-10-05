"""Stage 4 - Fitted preprocessing pipeline (fitted on TRAIN only).

    raw application rows
      -> ApplicationFeatureEngineer   (stateless feature engineering)
      -> TrainOnlyColumnFilter        (drop >95%-missing / constant TRAIN columns)
      -> ColumnTransformer
           num: median imputation + missing indicators
           ohe: one-hot, rare TRAIN categories grouped as "infrequent"
           te : cross-fitted target encoding (zip_code, emp_title, title)
      -> DropDuplicateColumns         (drop exact TRAIN duplicates)

Always fit with `fit_transform(X_train, y_train)`: that call produces the
cross-fitted (out-of-fold) target encodings for the training rows.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

from . import config
from .features import (
    ApplicationFeatureEngineer,
    DropDuplicateColumns,
    SelectNumericExcept,
    SelectPresent,
    TrainOnlyColumnFilter,
)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    one_hot = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=config.MISSING_CATEGORY)),
        ("encode", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=config.ONE_HOT_MIN_FREQUENCY,
            sparse_output=False,
            dtype=np.float64,
        )),
    ])
    target_encoded = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=config.MISSING_CATEGORY)),
        ("encode", TargetEncoder(
            target_type="binary",
            smooth="auto",
            cv=StratifiedKFold(
                n_splits=config.TARGET_ENCODER_CV,
                shuffle=True,
                random_state=config.RANDOM_STATE,
            ),
        )),
    ])

    encoder = ColumnTransformer(
        [
            ("num", numeric, SelectNumericExcept(config.CATEGORICAL_COLUMNS)),
            ("ohe", one_hot, SelectPresent(config.ONE_HOT_COLUMNS)),
            ("te", target_encoded, SelectPresent(config.TARGET_ENCODED_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    return Pipeline([
        ("engineer", ApplicationFeatureEngineer()),
        ("filter", TrainOnlyColumnFilter(max_missing_rate=config.MAX_TRAIN_MISSING_RATE)),
        ("encode", encoder),
        ("dedupe", DropDuplicateColumns()),
    ])
