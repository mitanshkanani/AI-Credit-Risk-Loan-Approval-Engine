"""Stage 7 - Final audits.

Every check is recorded as PASS / FAIL (required) or PASS / WARN (review
flag). Preprocessing V2 is complete only if no required check FAILS.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score

from . import config


class AuditLog:
    def __init__(self):
        self.results = []

    def record(self, category, check, passed, detail="", warning_only=False):
        if passed:
            status = "PASS"
        else:
            status = "WARN" if warning_only else "FAIL"
        self.results.append({
            "category": category,
            "check": check,
            "status": status,
            "detail": detail,
        })
        return passed

    @property
    def failures(self):
        return [r for r in self.results if r["status"] == "FAIL"]

    @property
    def warnings(self):
        return [r for r in self.results if r["status"] == "WARN"]

    def frame(self):
        return pd.DataFrame(self.results)


# ------------------------------------------------------------------
# Per-split checks (run while each transformed split is in memory)
# ------------------------------------------------------------------
def split_statistics(X, feature_names):
    """Checkable statistics for one transformed split (or one chunk of it)."""
    values = X.to_numpy()
    return {
        "rows": len(X),
        "nan": int(np.isnan(values).sum()),
        "inf": int(np.isinf(values).sum()),
        "non_numeric": sorted(c for c in X.columns if not is_numeric_dtype(X[c])),
        "duplicate_names": sorted(X.columns[X.columns.duplicated()].tolist()),
        "columns_match": list(X.columns) == list(feature_names),
        "n_columns": X.shape[1],
    }


def merge_statistics(parts):
    """Combine chunk statistics into split statistics."""
    return {
        "rows": sum(p["rows"] for p in parts),
        "nan": sum(p["nan"] for p in parts),
        "inf": sum(p["inf"] for p in parts),
        "non_numeric": sorted({c for p in parts for c in p["non_numeric"]}),
        "duplicate_names": sorted({c for p in parts for c in p["duplicate_names"]}),
        "columns_match": all(p["columns_match"] for p in parts),
        "n_columns": sorted({p["n_columns"] for p in parts}),
    }


def audit_split(log, split, stats, y_frame, feature_names):
    tag = f"[{split}]"
    n_columns = stats["n_columns"] if isinstance(stats["n_columns"], list) else [stats["n_columns"]]

    log.record("Missing values", f"{tag} no NaN in features",
               stats["nan"] == 0, f"NaN count = {stats['nan']}")
    log.record("Infinite values", f"{tag} no +/-inf in features",
               stats["inf"] == 0, f"inf count = {stats['inf']}")
    log.record("Non-numeric features", f"{tag} every feature is numeric",
               not stats["non_numeric"], f"non-numeric = {stats['non_numeric']}")
    log.record("Duplicate columns", f"{tag} no duplicate column names",
               not stats["duplicate_names"], f"duplicates = {stats['duplicate_names']}")
    log.record("Feature alignment", f"{tag} columns identical (names + order) to fitted feature list",
               stats["columns_match"], f"{n_columns} columns")
    log.record("Feature count", f"{tag} feature count matches fitted pipeline",
               n_columns == [len(feature_names)], f"{n_columns} vs {len(feature_names)}")

    target = y_frame[config.TARGET_COLUMN]
    log.record("Target alignment", f"{tag} X rows == y rows",
               stats["rows"] == len(y_frame), f"X {stats['rows']} / y {len(y_frame)}")
    log.record("Target alignment", f"{tag} target has no NaN and only {{0, 1}}",
               target.notna().all() and set(target.unique()) <= {0, 1},
               f"values = {sorted(target.unique().tolist())}")
    log.record("Target alignment", f"{tag} ids unique within split",
               y_frame["id"].is_unique, f"{y_frame['id'].nunique()} unique ids")


# ------------------------------------------------------------------
# Training-set checks
# ------------------------------------------------------------------
def duplicate_content_columns(X):
    by_hash = {}
    duplicates = []
    for column in X.columns:
        key = int(pd.util.hash_pandas_object(X[column], index=False).sum())
        for other in by_hash.get(key, []):
            if X[column].equals(X[other]):
                duplicates.append((column, other))
                break
        else:
            by_hash.setdefault(key, []).append(column)
    return duplicates


def single_feature_auc(X, y):
    rows = []
    for column in X.columns:
        auc = roc_auc_score(y, X[column])
        rows.append({"feature": column, "auc": auc, "auc_max_direction": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc_max_direction", ascending=False).reset_index(drop=True)


def audit_training_fit(log, preprocessor, X_train_raw, y_train, X_train_out):
    """Prove every fitted parameter came from the training split only."""
    engineered = preprocessor.named_steps["engineer"].transform(X_train_raw)
    filtered = preprocessor.named_steps["filter"].transform(engineered)
    encoder = preprocessor.named_steps["encode"]

    # Filter decisions are reproducible from training data alone.
    refit_drop = set()
    for column in engineered.columns:
        rate = engineered[column].isna().mean()
        if rate > config.MAX_TRAIN_MISSING_RATE or engineered[column].nunique(dropna=True) <= 1:
            refit_drop.add(column)
    log.record("Leakage (train-only fitting)", "column filter decisions reproduce from TRAIN",
               refit_drop == set(preprocessor.named_steps["filter"].dropped_),
               f"{len(refit_drop)} columns dropped")

    # Every engineered column is consumed by exactly one transformer.
    consumed = [
        column
        for name, _, columns in encoder.transformers_
        if name != "remainder"
        for column in columns
    ]
    log.record("Leakage (train-only fitting)", "every filtered column handled by a transformer (no silent remainder)",
               sorted(consumed) == sorted(filtered.columns) and len(consumed) == len(set(consumed)),
               f"{len(consumed)} consumed / {filtered.shape[1]} available")

    # Median imputer statistics equal TRAIN medians.
    num_cols = encoder.named_transformers_["num"].feature_names_in_
    imputer = encoder.named_transformers_["num"].named_steps["impute"]
    train_medians = filtered[list(num_cols)].median().to_numpy()
    log.record("Leakage (train-only fitting)", "numeric medians equal TRAIN medians",
               np.allclose(imputer.statistics_, train_medians, equal_nan=True),
               f"{len(num_cols)} numeric columns")

    # One-hot categories all come from TRAIN values.
    ohe_pipe = encoder.named_transformers_["ohe"]
    ohe = ohe_pipe.named_steps["encode"]
    unseen = {}
    for column, categories in zip(ohe_pipe.feature_names_in_, ohe.categories_):
        train_values = set(filtered[column].fillna(config.MISSING_CATEGORY).unique())
        extra = set(categories) - train_values
        if extra:
            unseen[column] = sorted(extra)
    log.record("Leakage (train-only fitting)", "one-hot categories are a subset of TRAIN values",
               not unseen, f"non-train categories = {unseen}")

    # Target encoder: prior equals TRAIN default rate; categories from TRAIN.
    te_pipe = encoder.named_transformers_["te"]
    te = te_pipe.named_steps["encode"]
    log.record("Leakage (train-only fitting)", "target-encoder prior equals TRAIN default rate",
               np.isclose(te.target_mean_, y_train.mean()),
               f"prior {te.target_mean_:.6f} vs train {y_train.mean():.6f}")
    te_cols = list(te_pipe.feature_names_in_)
    non_train = {}
    for column, categories in zip(te_cols, te.categories_):
        train_values = set(filtered[column].fillna(config.MISSING_CATEGORY).unique())
        extra = set(categories) - train_values
        if extra:
            non_train[column] = len(extra)
    log.record("Leakage (train-only fitting)", "target-encoder categories are a subset of TRAIN values",
               not non_train, f"non-train categories = {non_train}")

    # Cross-fitting: training-row encodings must be out-of-fold, i.e. differ
    # from what the fully-fitted encoder would give the same rows.
    full_fit = te_pipe.transform(filtered[te_cols])
    detail = {}
    ok = True
    for column in te_cols:
        differs = float((X_train_out[column].to_numpy() != full_fit[column].to_numpy()).mean())
        detail[column] = round(differs, 4)
        ok &= differs > 0.90
    log.record("Leakage (target encoding)", "TRAIN encodings are out-of-fold (cross-fitted), not self-encoded",
               ok, f"share of train rows whose OOF encoding differs from full-fit encoding = {detail}")


def audit_feature_names_for_leakage(log, feature_names):
    forbidden = set(config.FORBIDDEN_FEATURE_SOURCES)
    hits = []
    for name in feature_names:
        stripped = name.removeprefix("missingindicator_")
        if stripped in forbidden or any(stripped.startswith(f + "_") for f in forbidden if f not in {"id"}):
            hits.append(name)
    log.record("Leakage (feature names)", "no target / id / date / post-origination column among features",
               not hits, f"offending features = {hits}")


def audit_single_feature_auc(log, auc_table):
    flagged = auc_table[auc_table["auc_max_direction"] > config.SINGLE_FEATURE_AUC_REVIEW_THRESHOLD]
    top = auc_table.head(5)[["feature", "auc_max_direction"]].round(4).values.tolist()
    log.record("Leakage (single-feature AUC review)",
               f"no single feature has TRAIN AUC > {config.SINGLE_FEATURE_AUC_REVIEW_THRESHOLD} (review flag, not a failure)",
               flagged.empty,
               f"flagged = {flagged['feature'].tolist()}; top-5 = {top}",
               warning_only=True)


# ------------------------------------------------------------------
# Cross-split checks
# ------------------------------------------------------------------
def audit_temporal_ordering(log, y_frames):
    months = {s: pd.to_datetime(f["issue_d"], format="%b-%Y") for s, f in y_frames.items()}
    for split, (start, end) in config.SPLIT_WINDOWS.items():
        inside = months[split].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="left").all()
        log.record("Temporal ordering", f"[{split}] every issue month inside {start} .. {end} (exclusive)",
                   bool(inside), f"{months[split].min():%Y-%m} .. {months[split].max():%Y-%m}")
    order = list(config.SPLIT_WINDOWS)
    for earlier, later in zip(order, order[1:]):
        log.record("Temporal ordering", f"latest {earlier} month < earliest {later} month",
                   months[earlier].max() < months[later].min(),
                   f"{months[earlier].max():%Y-%m} < {months[later].min():%Y-%m}")
    all_ids = pd.concat([f["id"] for f in y_frames.values()])
    log.record("Temporal ordering", "no loan id appears in more than one split",
               all_ids.is_unique, f"{len(all_ids)} ids, {all_ids.duplicated().sum()} repeated")


def audit_target_spot_check(log, y_frames, rng_seed=config.RANDOM_STATE):
    """Independently re-read raw loan_status for a random sample of ids."""
    modeling = pd.concat([y_frames[s] for s in config.MODELING_SPLITS], ignore_index=True)
    sample = modeling.sample(n=config.TARGET_SPOT_CHECK_SIZE, random_state=rng_seed)
    wanted = set(sample["id"])
    found = {}
    for chunk in pd.read_csv(config.RAW_ACCEPTED_PATH, usecols=["id", "loan_status", "issue_d"],
                             dtype=str, chunksize=500_000):
        hit = chunk[chunk["id"].str.strip().isin(wanted)]
        for row in hit.itertuples(index=False):
            found[row.id.strip()] = (config.TARGET_MAPPING.get(row.loan_status), row.issue_d.strip())
    raw_target = sample["id"].map(lambda i: found.get(i, (None, None))[0])
    raw_issue = sample["id"].map(lambda i: found.get(i, (None, None))[1])
    target_ok = (raw_target.to_numpy() == sample[config.TARGET_COLUMN].to_numpy()).mean()
    issue_ok = (raw_issue.to_numpy() == sample["issue_d"].to_numpy()).mean()
    log.record("Target alignment", f"{config.TARGET_SPOT_CHECK_SIZE} random ids: target matches raw loan_status",
               target_ok == 1.0, f"match rate = {target_ok:.4f}")
    log.record("Target alignment", f"{config.TARGET_SPOT_CHECK_SIZE} random ids: issue_d matches raw",
               issue_ok == 1.0, f"match rate = {issue_ok:.4f}")
