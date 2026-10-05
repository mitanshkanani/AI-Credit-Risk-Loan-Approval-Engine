"""Stage 2 - Temporal splitting by loan issue month."""

import pandas as pd

from . import config


def assign_split(issue_d):
    """Label every row with its split; rows before Aug-2012 are excluded."""
    issue_month = pd.to_datetime(issue_d, format="%b-%Y")
    labels = pd.Series(config.EXCLUDED_LABEL, index=issue_d.index, dtype=object)
    for name, (start, end) in config.SPLIT_WINDOWS.items():
        in_window = (issue_month >= pd.Timestamp(start)) & (issue_month < pd.Timestamp(end))
        labels[in_window] = name
    return labels, issue_month


def split_summary(labels, issue_month, target):
    """Exact rows, defaults, default rate and date range per split."""
    frame = pd.DataFrame({"split": labels, "issue_month": issue_month, "target": target})
    order = list(config.SPLIT_WINDOWS) + [config.EXCLUDED_LABEL]
    summary = (
        frame.groupby("split")
        .agg(
            rows=("target", "size"),
            defaults=("target", "sum"),
            default_rate=("target", "mean"),
            first_issue_month=("issue_month", "min"),
            last_issue_month=("issue_month", "max"),
        )
        .reindex(order)
    )
    summary["defaults"] = summary["defaults"].astype(int)
    summary["first_issue_month"] = summary["first_issue_month"].dt.strftime("%Y-%m")
    summary["last_issue_month"] = summary["last_issue_month"].dt.strftime("%Y-%m")
    return summary
