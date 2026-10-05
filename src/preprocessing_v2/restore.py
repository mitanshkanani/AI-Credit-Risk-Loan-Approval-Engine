"""Stage 1 - Data restoration.

Re-attaches `id`, `issue_d` and `target` to the Step 10 dataset
(accepted_quality_clean.csv), which does not contain them.

Safety: the raw CSV is re-read in its original order, filtered with the
same Step 3 target mapping, and 12 shared columns are compared row by row
with the Step 10 dataset. Any row-count difference or any single
mismatching value stops the run before anything is attached.
"""

import numpy as np
import pandas as pd

from . import config


def _normalize_numeric(values):
    text = values.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(text, errors="coerce").astype("float64").round(6)


def _normalize_text(values):
    # Same storage on both sides (raw = object, base = pyarrow) so hashes compare values only.
    return values.astype("string[python]").str.strip().fillna("<NA>")


def _column_hashes(frame):
    hashes = {}
    for column in config.ALIGNMENT_NUMERIC_COLUMNS:
        hashes[column] = pd.util.hash_pandas_object(
            _normalize_numeric(frame[column]), index=False).to_numpy()
    for column in config.ALIGNMENT_TEXT_COLUMNS:
        hashes[column] = pd.util.hash_pandas_object(
            _normalize_text(frame[column]), index=False).to_numpy()
    return hashes


def read_raw_eligible_keys(chunksize=250_000):
    """Read id / issue_d / target (+ alignment hashes) for eligible raw rows, in order."""
    alignment_columns = config.ALIGNMENT_NUMERIC_COLUMNS + config.ALIGNMENT_TEXT_COLUMNS
    usecols = ["id", "loan_status", "issue_d"] + alignment_columns

    key_parts = []
    hash_parts = {c: [] for c in alignment_columns}
    raw_rows = 0
    for chunk in pd.read_csv(config.RAW_ACCEPTED_PATH, usecols=usecols,
                             dtype=str, chunksize=chunksize):
        raw_rows += len(chunk)
        target = chunk["loan_status"].map(config.TARGET_MAPPING)
        eligible = chunk.loc[target.notna()]
        key_parts.append(pd.DataFrame({
            "id": eligible["id"].str.strip().to_numpy(),
            "issue_d": eligible["issue_d"].str.strip().to_numpy(),
            "target": target[target.notna()].astype("int8").to_numpy(),
        }))
        for column, values in _column_hashes(eligible).items():
            hash_parts[column].append(values)

    keys = pd.concat(key_parts, ignore_index=True).astype(
        {"id": "string[pyarrow]", "issue_d": "string[pyarrow]"})
    hashes = {c: np.concatenate(parts) for c, parts in hash_parts.items()}
    return keys, hashes, raw_rows


def load_base_dataset():
    """Load the Step 10 dataset, skipping by-design exclusions (desc)."""
    skip = set(config.DROPPED_BY_DESIGN)
    header = pd.read_csv(config.BASE_DATASET_PATH, nrows=0).columns
    text_dtypes = {c: "string[pyarrow]" for c in config.TEXT_INPUT_COLUMNS if c in header}
    return pd.read_csv(config.BASE_DATASET_PATH, low_memory=False,
                       usecols=lambda c: c not in skip, dtype=text_dtypes)


def restore_keys_and_target():
    """Return the Step 10 dataset with id / issue_d / target restored, plus a log."""
    keys, raw_hashes, raw_rows = read_raw_eligible_keys()
    base = load_base_dataset()

    log = {
        "raw_rows": int(raw_rows),
        "raw_eligible_rows": int(len(keys)),
        "base_rows": int(len(base)),
        "base_columns_loaded": int(base.shape[1]),
    }
    if len(keys) != len(base):
        raise ValueError(
            f"Row counts differ (raw eligible {len(keys)} vs base {len(base)}); "
            "restoration by position is unsafe."
        )

    base_hashes = _column_hashes(base)
    mismatches = {c: int((raw_hashes[c] != base_hashes[c]).sum()) for c in raw_hashes}
    log["alignment_columns_checked"] = list(mismatches)
    log["alignment_mismatches"] = mismatches
    if any(mismatches.values()):
        raise ValueError(f"Row alignment check failed: {mismatches}")

    if not keys["id"].is_unique or keys["id"].isna().any():
        raise ValueError("Raw `id` values are not unique / complete.")
    issue_month = pd.to_datetime(keys["issue_d"], format="%b-%Y", errors="coerce")
    if issue_month.isna().any():
        raise ValueError("Some `issue_d` values could not be parsed.")

    base.insert(0, "id", keys["id"].array)
    base.insert(1, "issue_d", keys["issue_d"].array)
    base[config.TARGET_COLUMN] = keys["target"].to_numpy()
    log["id_unique"] = True
    log["issue_d_parse_failures"] = 0
    log["issue_d_range"] = [str(issue_month.min().date()), str(issue_month.max().date())]
    log["target_counts"] = {
        str(k): int(v)
        for k, v in base[config.TARGET_COLUMN].value_counts().sort_index().items()
    }
    return base, log
