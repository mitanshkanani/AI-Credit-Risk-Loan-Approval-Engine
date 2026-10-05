"""Shared report card for every model (M0-M5).

All models are scored with exactly the same functions so their numbers are
directly comparable in models/leaderboard.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "final_preprocessed_data_v2"
METADATA_PATH = PROJECT_ROOT / "artifacts_v2" / "metadata_v2.json"
LEADERBOARD_PATH = PROJECT_ROOT / "models" / "leaderboard.csv"

# The test split stays sealed until the final model is chosen.
OPEN_SPLITS = ("train", "validation")


# ------------------------------------------------------------------ data
def load_split(split, feature_columns=None, allow_test=False):
    """Return (X, y_frame). X holds only `feature_columns` ([] -> empty frame, None -> all)."""
    if split == "test" and not allow_test:
        raise PermissionError("The test split is sealed until the final model is chosen.")
    y_frame = pd.read_parquet(DATA_DIR / f"y_{split}.parquet")
    if feature_columns is not None and len(feature_columns) == 0:
        X = pd.DataFrame(index=y_frame.index)
    else:
        X = pd.read_parquet(DATA_DIR / f"X_{split}.parquet", columns=feature_columns)
    if len(X) != len(y_frame):
        raise ValueError(f"{split}: X has {len(X)} rows but y has {len(y_frame)}")
    return X, y_frame


# ------------------------------------------------------------------ metrics
def ks_statistic(y_true, pd_scores):
    """Largest gap between the cumulative score distributions of defaulters and payers."""
    fpr, tpr, _ = roc_curve(y_true, pd_scores)
    return float(np.max(tpr - fpr))


def report_card(y_true, pd_scores, threshold=0.5):
    """Standard metrics for one set of predictions."""
    y_true = np.asarray(y_true).astype(int)
    pd_scores = np.clip(np.asarray(pd_scores, dtype=float), 1e-15, 1 - 1e-15)
    predicted_default = pd_scores >= threshold
    actual_rate = y_true.mean()
    return {
        "rows": int(len(y_true)),
        "defaults": int(y_true.sum()),
        "default_rate": float(actual_rate),
        "mean_pd": float(pd_scores.mean()),
        "calibration_gap": float(pd_scores.mean() - actual_rate),
        "roc_auc": float(roc_auc_score(y_true, pd_scores)),
        "pr_auc": float(average_precision_score(y_true, pd_scores)),
        "ks": ks_statistic(y_true, pd_scores),
        "brier": float(brier_score_loss(y_true, pd_scores)),
        "log_loss": float(log_loss(y_true, pd_scores, labels=[0, 1])),
        "accuracy_at_threshold": float((predicted_default == y_true).mean()),
        "default_recall_at_threshold": float(predicted_default[y_true == 1].mean()) if y_true.any() else 0.0,
        "threshold": threshold,
    }


def segment_report(y_true, pd_scores, segment, segment_name):
    """report_card per segment value (e.g. term_months 36 / 60)."""
    rows = []
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(pd_scores), "s": np.asarray(segment)})
    for value, part in frame.groupby("s"):
        card = report_card(part["y"], part["p"])
        rows.append({segment_name: value, **card})
    return pd.DataFrame(rows)


def approval_curve(y_true, pd_scores, approval_rates=None):
    """Bad rate among the approved when approving the safest X% (lowest PD).

    Applicants with identical PD are treated as a block: when the cut-off falls
    inside a block, the block's average bad rate is used (expected value), so a
    model that cannot rank (e.g. a constant PD) gets the overall bad rate.
    """
    if approval_rates is None:
        approval_rates = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    blocks = (
        pd.DataFrame({"p": np.asarray(pd_scores, dtype=float), "y": np.asarray(y_true, dtype=int)})
        .groupby("p")["y"].agg(["size", "sum"]).sort_index()
    )
    cum_n = blocks["size"].cumsum().to_numpy()
    cum_bad = blocks["sum"].cumsum().to_numpy()
    sizes, bads = blocks["size"].to_numpy(), blocks["sum"].to_numpy()
    total = cum_n[-1]
    rows = []
    for rate in approval_rates:
        k = rate * total
        i = int(np.searchsorted(cum_n, k, side="left"))
        before_n = cum_n[i - 1] if i > 0 else 0
        before_bad = cum_bad[i - 1] if i > 0 else 0
        partial = k - before_n
        bad = before_bad + partial * bads[i] / sizes[i]
        rows.append({"approval_rate": float(rate), "approved": int(round(k)),
                     "bad_rate_among_approved": float(bad / k) if k else 0.0})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ leaderboard
def update_leaderboard(model, version, split, card, notes=""):
    """Insert or replace the row for (model, version, split) in models/leaderboard.csv."""
    row = {"model": model, "version": version, "split": split,
           **{k: card[k] for k in ("rows", "default_rate", "mean_pd", "roc_auc", "pr_auc",
                                   "ks", "brier", "log_loss", "calibration_gap")},
           "notes": notes}
    if LEADERBOARD_PATH.exists():
        board = pd.read_csv(LEADERBOARD_PATH)
        key = (board["model"] == model) & (board["version"] == version) & (board["split"] == split)
        board = pd.concat([board.loc[~key], pd.DataFrame([row])], ignore_index=True)
    else:
        board = pd.DataFrame([row])
    board = board.sort_values(["model", "version", "split"]).reset_index(drop=True)
    LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(LEADERBOARD_PATH, index=False)
    return board
