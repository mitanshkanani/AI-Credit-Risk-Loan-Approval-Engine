"""Load the saved V2 preprocessor and transform new raw loan applications."""

import joblib
import numpy as np
import pandas as pd

from . import config


def load_preprocessor(path=config.PREPROCESSOR_PATH):
    return joblib.load(path)


def required_input_columns(preprocessor):
    """Raw columns the preprocessor expects (`issue_d` = application month)."""
    return list(preprocessor.named_steps["engineer"].input_columns_)


def build_application_frame(records, preprocessor):
    """Build a raw-input frame; fields not supplied are treated as missing."""
    frame = pd.DataFrame(records)
    return frame.reindex(columns=required_input_columns(preprocessor), fill_value=np.nan)


def transform_applications(records, preprocessor):
    return preprocessor.transform(build_application_frame(records, preprocessor))
