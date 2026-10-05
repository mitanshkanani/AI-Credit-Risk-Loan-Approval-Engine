"""Feature engineering and custom transformers for preprocessing V2.

Everything stored inside the saved preprocessor is defined here at module
level, so joblib can re-import it in a completely fresh Python session.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin

from . import config

EMP_LENGTH_YEARS = {
    "< 1 year": 0.0,
    "1 year": 1.0,
    "2 years": 2.0,
    "3 years": 3.0,
    "4 years": 4.0,
    "5 years": 5.0,
    "6 years": 6.0,
    "7 years": 7.0,
    "8 years": 8.0,
    "9 years": 9.0,
    "10+ years": 10.0,
}


# ------------------------------------------------------------------
# Row-level feature engineering (no fitted state)
# ------------------------------------------------------------------
def parse_month(values):
    """Parse 'Mon-YYYY' strings (or datetimes) into month timestamps."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    text = pd.Series(values, copy=False).astype("string").str.strip()
    return pd.to_datetime(text, format="%b-%Y", errors="coerce")


def months_between(later, earlier):
    """Whole calendar months from `earlier` to `later` (month precision)."""
    return (
        (later.dt.year - earlier.dt.year) * 12
        + (later.dt.month - earlier.dt.month)
    ).astype("float64")


def _clean_category(values, lowercase):
    text = values.astype("string").str.strip()
    if lowercase:
        text = text.str.lower()
    text = text.mask(text == "")
    return text.astype(object).where(text.notna(), np.nan)


def _to_number(values):
    if is_numeric_dtype(values):
        return values.astype("float64")
    text = values.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(text, errors="coerce").astype("float64")


def engineer_features(frame):
    """Turn one or more raw application rows into model-input columns.

    - credit_history_months = (app_year - ecl_year) * 12 + (app_month - ecl_month)
    - term_months / emp_length_years: numeric versions of text fields
    - categorical/text fields are stripped (target-encoded ones lowercased)
    - every other column is coerced to float
    - raw dates, keys and by-design exclusions are dropped
    """
    out = frame.copy()
    application_month = parse_month(out[config.APPLICATION_DATE_COLUMN])

    out["credit_history_months"] = months_between(
        application_month, parse_month(out["earliest_cr_line"])
    )
    if "sec_app_earliest_cr_line" in out.columns:
        out["sec_app_credit_history_months"] = months_between(
            application_month, parse_month(out["sec_app_earliest_cr_line"])
        )

    out["term_months"] = pd.to_numeric(
        out["term"].astype("string").str.extract(r"(\d+)", expand=False),
        errors="coerce",
    ).astype("float64")
    out["emp_length_years"] = (
        out["emp_length"].astype("string").str.strip()
        .map(EMP_LENGTH_YEARS).astype("float64")
    )

    to_drop = (
        list(config.REPLACED_BY_ENGINEERING)
        + list(config.DROPPED_BY_DESIGN)
        + config.KEY_COLUMNS
        + [config.TARGET_COLUMN]
    )
    out = out.drop(columns=[c for c in to_drop if c in out.columns])

    for column in out.columns:
        if column in config.CATEGORICAL_COLUMNS:
            out[column] = _clean_category(
                out[column], lowercase=column in config.TARGET_ENCODED_COLUMNS
            )
        else:
            out[column] = _to_number(out[column])
    return out


class ApplicationFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless wrapper so feature engineering lives inside the pipeline."""

    def fit(self, X, y=None):
        self.input_columns_ = list(X.columns)
        return self

    def transform(self, X):
        missing = [c for c in self.input_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Application is missing input columns: {missing}")
        return engineer_features(X[self.input_columns_])


# ------------------------------------------------------------------
# Fitted transformers (state learned from TRAIN only)
# ------------------------------------------------------------------
class TrainOnlyColumnFilter(BaseEstimator, TransformerMixin):
    """Drop columns that are almost always missing or constant in TRAIN."""

    def __init__(self, max_missing_rate=0.95):
        self.max_missing_rate = max_missing_rate

    def fit(self, X, y=None):
        missing_rate = X.isna().mean()
        unique_count = X.nunique(dropna=True)
        self.dropped_ = {}
        for column in X.columns:
            if missing_rate[column] > self.max_missing_rate:
                self.dropped_[column] = (
                    f"Missing in {missing_rate[column]:.1%} of training rows "
                    f"(> {self.max_missing_rate:.0%})"
                )
            elif unique_count[column] <= 1:
                self.dropped_[column] = "Constant (single value) in training rows"
        self.kept_columns_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X):
        missing = [c for c in self.kept_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Columns expected by the filter are missing: {missing}")
        return X[self.kept_columns_].copy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


class DropDuplicateColumns(BaseEstimator, TransformerMixin):
    """Drop output columns whose TRAIN values exactly duplicate an earlier column.

    Typical case: several missing-value indicators that share one pattern.
    """

    def fit(self, X, y=None):
        self.duplicates_ = {}
        first_by_hash = {}
        for column in X.columns:
            key = int(pd.util.hash_pandas_object(X[column], index=False).sum())
            for kept in first_by_hash.get(key, []):
                if X[column].equals(X[kept]):
                    self.duplicates_[column] = kept
                    break
            else:
                first_by_hash.setdefault(key, []).append(column)
        self.kept_columns_ = [c for c in X.columns if c not in self.duplicates_]
        return self

    def transform(self, X):
        return X[self.kept_columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.kept_columns_, dtype=object)


# ------------------------------------------------------------------
# Column selectors for ColumnTransformer (picklable callables)
# ------------------------------------------------------------------
class SelectPresent:
    """Select the listed columns that survived the train-only filter."""

    def __init__(self, names):
        self.names = list(names)

    def __call__(self, X):
        return [c for c in self.names if c in X.columns]


class SelectNumericExcept:
    """Select numeric columns that are not in an excluded group."""

    def __init__(self, exclude):
        self.exclude = list(exclude)

    def __call__(self, X):
        return [
            c for c in X.columns
            if c not in self.exclude and is_numeric_dtype(X[c])
        ]
