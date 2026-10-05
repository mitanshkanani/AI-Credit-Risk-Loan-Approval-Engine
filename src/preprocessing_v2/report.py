"""Human-readable outputs: removed / engineered feature tables and a markdown report."""

import json

import pandas as pd

from . import config


def removed_features_table(ctx):
    rows = []
    step7 = pd.read_csv(config.PROJECT_ROOT / "accepted_step7_removal_audit.csv")
    for row in step7.itertuples(index=False):
        rows.append(("V1 Step 7 (carried over)", row.column, row.reason))
    for column, reason in config.DROPPED_BY_DESIGN.items():
        rows.append(("V2 by design", column, reason))
    for column, new in config.REPLACED_BY_ENGINEERING.items():
        rows.append(("V2 replaced by engineered feature", column, f"Replaced by `{new}`"))
    rows.append(("V2 key column", "issue_d",
                 "Restored only for the temporal split and credit_history_months; never a model feature"))
    pre = ctx["preprocessor"]
    for column, reason in pre.named_steps["filter"].dropped_.items():
        rows.append(("V2 train-only filter", column, reason))
    for column, kept in pre.named_steps["dedupe"].duplicates_.items():
        rows.append(("V2 duplicate content (train)", column, f"Identical to `{kept}` in training rows"))
    return pd.DataFrame(rows, columns=["stage", "column", "reason"])


def engineered_features_table(ctx):
    names = ctx["feature_names"]
    pre = ctx["preprocessor"]
    dropped = pre.named_steps["filter"].dropped_
    rows = []
    formulas = {
        "credit_history_months": "(issue_year - ecl_year) * 12 + (issue_month - ecl_month), from earliest_cr_line",
        "sec_app_credit_history_months": "Same formula from sec_app_earliest_cr_line",
        "term_months": "Numeric months parsed from `term` (36 / 60)",
        "emp_length_years": "`< 1 year` -> 0 ... `10+ years` -> 10",
    }
    for feature, how in formulas.items():
        status = "in final features" if feature in names else f"dropped: {dropped.get(feature, 'n/a')}"
        rows.append(("Row-level engineering", feature, how, status))
    for feature in names:
        if feature.startswith("missingindicator_"):
            rows.append(("Missing-value indicator", feature,
                         f"1 if `{feature.removeprefix('missingindicator_')}` was missing (median-imputed)",
                         "in final features"))
    ohe_sources = ctx["metadata"]["feature_groups"]["one_hot_sources"]
    for source in ohe_sources:
        produced = [f for f in names if f.startswith(source + "_")]
        rows.append(("One-hot encoding", f"{source}_*", f"{len(produced)} columns (rare train categories -> infrequent)",
                     "in final features"))
    for source in ctx["metadata"]["feature_groups"]["target_encoded"]:
        rows.append(("Target encoding (cross-fitted)", source,
                     "Smoothed training default rate of the category; out-of-fold for train rows",
                     "in final features"))
    return pd.DataFrame(rows, columns=["type", "feature", "definition", "status"])


def _markdown_table(frame):
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider] + body)


def write_reports(ctx):
    removed = removed_features_table(ctx)
    engineered = engineered_features_table(ctx)
    removed.to_csv(config.REMOVED_FEATURES_PATH, index=False)
    engineered.to_csv(config.ENGINEERED_FEATURES_PATH, index=False)

    summary = ctx["split_summary"].copy()
    summary["default_rate"] = (summary["default_rate"] * 100).round(2).astype(str) + "%"
    summary = summary.reset_index().rename(columns={"index": "split"})
    audit = ctx["audit"].frame()
    fresh = json.loads(config.FRESH_SESSION_RESULT_PATH.read_text(encoding="utf-8")) \
        if config.FRESH_SESSION_RESULT_PATH.exists() else {"passed": False, "checks": []}

    status = "COMPLETE - every required audit passed" if ctx["complete"] \
        else f"NOT COMPLETE - {len(ctx['audit'].failures)} required audit(s) failed"
    paths = [("Preprocessor", config.PREPROCESSOR_PATH), ("Metadata", config.METADATA_PATH),
             ("Audit report", config.AUDIT_REPORT_PATH), ("Feature list", config.FEATURE_LIST_PATH),
             ("Removed features", config.REMOVED_FEATURES_PATH),
             ("Engineered features", config.ENGINEERED_FEATURES_PATH),
             ("Single-feature AUC", config.SINGLE_FEATURE_AUC_PATH),
             ("Fresh-session result", config.FRESH_SESSION_RESULT_PATH)]
    rel = lambda p: str(p.relative_to(config.PROJECT_ROOT)).replace("\\", "/")  # noqa: E731

    lines = [
        "# Accepted-loan preprocessing V2 report", "",
        f"**Status:** {status}", "",
        "## Split summary", "", _markdown_table(summary), "",
        f"Final feature count: **{len(ctx['feature_names'])}**", "",
        "## Saved artifacts", "",
        *[f"- {label}: `{rel(path)}`" for label, path in paths], "",
        "## Saved datasets", "",
        *[f"- {split}: `{rel(config.PROJECT_ROOT / p['X'])}`, `{rel(config.PROJECT_ROOT / p['y'])}`"
          for split, p in ctx["dataset_paths"].items()], "",
        "## Removed features", "", _markdown_table(removed), "",
        "## Engineered features", "", _markdown_table(engineered), "",
        "## Audit results", "", _markdown_table(audit), "",
        "## Fresh-session inference test", "",
        f"Result: **{'PASSED' if fresh['passed'] else 'FAILED'}**", "",
        _markdown_table(pd.DataFrame(fresh["checks"])) if fresh["checks"] else "_no result file_", "",
    ]
    config.SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return config.SUMMARY_REPORT_PATH
