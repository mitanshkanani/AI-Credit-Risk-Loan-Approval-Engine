"""Preprocessing V2 orchestration.

Stages (each one is a function so the notebook can run them one by one):

    1. stage_restore              data restoration (id / issue_d / target)
    2. stage_split                temporal splitting
    3. stage_feature_engineering  engineered-feature preview + checks
    4. stage_fit                  fitted preprocessing (TRAIN only)
    5. stage_save_artifacts       preprocessor + metadata
    6. stage_export               transform + export every split
    7. stage_final_audit          cross-split audits, fresh-session test, report

`ctx` is a plain dict passed between stages.
"""

import gc
import subprocess
import sys
import time

import joblib
import pandas as pd
import pyarrow.parquet as pq

from . import audits, config, export
from .features import engineer_features
from .pipeline import build_preprocessor
from .restore import restore_keys_and_target
from .split import assign_split, split_summary


def _log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def new_context():
    return {"audit": audits.AuditLog(), "protected_before": export.snapshot_protected_paths()}


# ------------------------------------------------------------------
# 1. Data restoration
# ------------------------------------------------------------------
def stage_restore(ctx):
    _log("Stage 1 - restoring id / issue_d / target from the raw CSV")
    base, restore_log = restore_keys_and_target()
    ctx["base"] = base
    ctx["restore_log"] = restore_log
    log = ctx["audit"]
    log.record("Target alignment", "raw eligible row count == Step 10 row count",
               restore_log["raw_eligible_rows"] == restore_log["base_rows"],
               f"{restore_log['raw_eligible_rows']} == {restore_log['base_rows']}")
    log.record("Target alignment", "12 shared columns identical row-by-row (raw vs Step 10)",
               not any(restore_log["alignment_mismatches"].values()),
               str(restore_log["alignment_mismatches"]))
    _log(f"  restored {len(base):,} rows; alignment mismatches = {restore_log['alignment_mismatches']}")
    return ctx


# ------------------------------------------------------------------
# 2. Temporal splitting
# ------------------------------------------------------------------
def stage_split(ctx):
    _log("Stage 2 - temporal split by issue month")
    base = ctx["base"]
    labels, issue_month = assign_split(base["issue_d"])
    summary = split_summary(labels, issue_month, base[config.TARGET_COLUMN])
    ctx["split_summary"] = summary
    ctx["excluded_rows"] = int((labels == config.EXCLUDED_LABEL).sum())

    # Stage each split's raw rows on disk and free the full table, so later
    # stages only ever hold the split they are working on.
    export.prepare_output_dirs()
    for split in config.SPLIT_WINDOWS:
        export.stage_raw_split(split, base.loc[labels == split])
    del ctx["base"], base, labels
    gc.collect()
    print(summary.to_string())
    return ctx


# ------------------------------------------------------------------
# 3. Feature engineering (preview on TRAIN; the real work runs inside the pipeline)
# ------------------------------------------------------------------
def stage_feature_engineering(ctx):
    _log("Stage 3 - feature engineering preview (train rows)")
    train_raw = export.load_raw_split("train").drop(columns=[config.TARGET_COLUMN])
    engineered = engineer_features(train_raw)
    history = engineered["credit_history_months"]
    preview = {
        "input_columns": int(train_raw.shape[1]),
        "engineered_columns": int(engineered.shape[1]),
        "credit_history_months": {
            "missing": int(history.isna().sum()),
            "negative": int((history < 0).sum()),
            "min": float(history.min()),
            "median": float(history.median()),
            "max": float(history.max()),
        },
        "term_months_values": sorted(engineered["term_months"].dropna().unique().tolist()),
        "emp_length_years_missing": int(engineered["emp_length_years"].isna().sum()),
        "emp_length_unmapped": int(
            (train_raw["emp_length"].notna() & engineered["emp_length_years"].isna()).sum()
        ),
    }
    ctx["engineering_preview"] = preview
    log = ctx["audit"]
    log.record("Feature engineering", "credit_history_months has no negative values (TRAIN)",
               preview["credit_history_months"]["negative"] == 0, str(preview["credit_history_months"]))
    log.record("Feature engineering", "term parsed to {36, 60} months",
               set(preview["term_months_values"]) <= {36.0, 60.0}, str(preview["term_months_values"]))
    log.record("Feature engineering", "every non-missing emp_length mapped to years",
               preview["emp_length_unmapped"] == 0, f"unmapped = {preview['emp_length_unmapped']}")
    print(preview)
    del train_raw, engineered
    gc.collect()
    return ctx


# ------------------------------------------------------------------
# 4. Fitted preprocessing (TRAIN only)
# ------------------------------------------------------------------
def _split_frames(ctx, split):
    rows = export.load_raw_split(split)
    X_raw = rows.drop(columns=["id", config.TARGET_COLUMN])
    y_frame = rows[["id", "issue_d", config.TARGET_COLUMN]].reset_index(drop=True)
    return X_raw, y_frame


def stage_fit(ctx):
    _log("Stage 4 - fitting the preprocessor on TRAIN only")
    X_raw, y_frame = _split_frames(ctx, "train")
    y = y_frame[config.TARGET_COLUMN]

    preprocessor = build_preprocessor()
    # fit_transform -> out-of-fold target encodings for training rows
    X_train = preprocessor.fit_transform(X_raw, y)
    feature_names = list(X_train.columns)
    ctx.update(preprocessor=preprocessor, feature_names=feature_names,
               input_columns=list(X_raw.columns))
    _log(f"  fitted: {X_raw.shape[1]} raw input columns -> {len(feature_names)} features")

    log = ctx["audit"]
    audits.audit_training_fit(log, preprocessor, X_raw, y, X_train)
    duplicates = audits.duplicate_content_columns(X_train)
    log.record("Duplicate columns", "[train] no two features have identical values",
               not duplicates, f"duplicate pairs = {duplicates}")
    auc_table = audits.single_feature_auc(X_train, y)
    auc_table.to_csv(config.SINGLE_FEATURE_AUC_PATH, index=False)
    audits.audit_single_feature_auc(log, auc_table)
    audits.audit_feature_names_for_leakage(log, feature_names)

    del X_raw
    gc.collect()
    ctx["train_cache"] = (X_train, y_frame)
    return ctx


# ------------------------------------------------------------------
# 5. Artifact saving
# ------------------------------------------------------------------
def _lc_risk_features(feature_names):
    return [
        f for f in feature_names
        if f in ("int_rate", "installment")
        or f.startswith("grade_") or f.startswith("sub_grade_")
        or f in ("missingindicator_int_rate", "missingindicator_installment")
    ]


def build_metadata(ctx):
    pre = ctx["preprocessor"]
    encoder = pre.named_steps["encode"]
    ohe_pipe = encoder.named_transformers_["ohe"]
    te_pipe = encoder.named_transformers_["te"]
    summary = ctx["split_summary"].reset_index().rename(columns={"index": "split"})
    return {
        "version": "preprocessing_v2",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "environment": export.environment_info(),
        "source_files": {
            "raw": str(config.RAW_ACCEPTED_PATH.relative_to(config.PROJECT_ROOT)),
            "step10_base": str(config.BASE_DATASET_PATH.relative_to(config.PROJECT_ROOT)),
        },
        "split_windows": config.SPLIT_WINDOWS,
        "split_summary": summary.to_dict(orient="records"),
        "excluded_pre_2012_08_rows": ctx["excluded_rows"],
        "input_columns": ctx["input_columns"],
        "application_date_column": config.APPLICATION_DATE_COLUMN,
        "n_features": len(ctx["feature_names"]),
        "feature_names": ctx["feature_names"],
        "feature_groups": {
            "lc_risk_features": _lc_risk_features(ctx["feature_names"]),
            "target_encoded": list(te_pipe.feature_names_in_),
            "one_hot_sources": list(ohe_pipe.feature_names_in_),
            "missing_indicators": [f for f in ctx["feature_names"] if f.startswith("missingindicator_")],
        },
        "filter_dropped": pre.named_steps["filter"].dropped_,
        "duplicate_columns_dropped": pre.named_steps["dedupe"].duplicates_,
        "parameters": {
            "max_train_missing_rate": config.MAX_TRAIN_MISSING_RATE,
            "one_hot_min_frequency": config.ONE_HOT_MIN_FREQUENCY,
            "target_encoder_cv": config.TARGET_ENCODER_CV,
            "random_state": config.RANDOM_STATE,
        },
        "engineering_preview": ctx.get("engineering_preview"),
        "restore_log": ctx["restore_log"],
    }


def stage_save_artifacts(ctx):
    _log("Stage 5 - saving preprocessor and metadata")
    ctx["preprocessor_path"] = export.save_preprocessor(ctx["preprocessor"])
    ctx["metadata"] = build_metadata(ctx)
    export.save_json(ctx["metadata"], config.METADATA_PATH)
    pd.DataFrame({"position": range(len(ctx["feature_names"])), "feature": ctx["feature_names"]}) \
        .to_csv(config.FEATURE_LIST_PATH, index=False)
    _log(f"  {config.PREPROCESSOR_PATH}")
    return ctx


# ------------------------------------------------------------------
# 6. Export
# ------------------------------------------------------------------
def stage_export(ctx):
    _log("Stage 6 - transforming and exporting every split")
    log = ctx["audit"]
    pre, names = ctx["preprocessor"], ctx["feature_names"]
    ctx["y_frames"], ctx["dataset_paths"] = {}, {}

    for split in config.SPLIT_WINDOWS:
        if split == "train":
            X, y_frame = ctx.pop("train_cache")
            stats = audits.split_statistics(X, names)
            x_path, y_path = export.export_split(split, X, y_frame)
            del X
        else:
            X_raw, y_frame = _split_frames(ctx, split)
            chunk_stats = []
            step = config.TRANSFORM_CHUNK_ROWS
            chunks = (pre.transform(X_raw.iloc[start:start + step])
                      for start in range(0, len(X_raw), step))
            x_path, y_path = export.export_split_in_chunks(
                split, chunks, y_frame,
                on_chunk=lambda X: chunk_stats.append(audits.split_statistics(X, names)))
            stats = audits.merge_statistics(chunk_stats)
            del X_raw
        audits.audit_split(log, split, stats, y_frame, names)

        # read back what was written
        x_file_rows = pq.read_metadata(x_path).num_rows
        y_back = pd.read_parquet(y_path)
        log.record("Target alignment", f"[{split}] exported X and y files have equal rows and y matches memory",
                   x_file_rows == len(y_back) == stats["rows"] and y_back.equals(y_frame.reset_index(drop=True)),
                   f"X file rows {x_file_rows}, y file rows {len(y_back)}")
        ctx["y_frames"][split] = y_frame
        ctx["dataset_paths"][split] = {"X": str(x_path), "y": str(y_path)}
        _log(f"  {split}: X ({stats['rows']:,} x {len(names)}) -> {x_path.name}, y -> {y_path.name}")
        gc.collect()

    # Persistence: reload the saved file and compare on validation rows.
    reloaded = joblib.load(config.PREPROCESSOR_PATH)
    X_raw, _ = _split_frames(ctx, "validation")
    sample = X_raw.head(2000)
    same = reloaded.transform(sample).equals(pre.transform(sample))
    log.record("Fitted-transformer persistence", "reloaded preprocessor output identical to in-memory output",
               same, "2,000 validation rows")
    return ctx


# ------------------------------------------------------------------
# 7. Final audits + report
# ------------------------------------------------------------------
def _run_fresh_session_test():
    script = config.PROJECT_ROOT / "tests" / "fresh_session_inference_test.py"
    completed = subprocess.run([sys.executable, str(script)], cwd=config.PROJECT_ROOT,
                               capture_output=True, text=True)
    return completed


def stage_final_audit(ctx):
    _log("Stage 7 - final audits")
    log = ctx["audit"]
    audits.audit_temporal_ordering(log, ctx["y_frames"])
    audits.audit_target_spot_check(log, ctx["y_frames"])

    missing = [str(f) for f in [config.PREPROCESSOR_PATH, config.METADATA_PATH] if not f.exists()]
    log.record("Fitted-transformer persistence", "preprocessor_v2.joblib and metadata_v2.json exist",
               not missing, f"missing = {missing}")

    schemas = {split: pq.read_schema(paths["X"]).names for split, paths in ctx["dataset_paths"].items()}
    counts = {split: len(names) for split, names in schemas.items()}
    consistent = (
        set(counts.values()) == {ctx["metadata"]["n_features"]}
        and all(names == ctx["feature_names"] for names in schemas.values())
    )
    log.record("Feature count", "every exported X file has the metadata feature count and order",
               consistent, f"{counts} vs metadata {ctx['metadata']['n_features']}")

    lc = ctx["metadata"]["feature_groups"]["lc_risk_features"]
    log.record("Modeling readiness", "LC risk features (grade, sub_grade, int_rate, installment) available for A/B",
               any(f.startswith("grade_") for f in lc) and any(f.startswith("sub_grade_") for f in lc)
               and "int_rate" in lc and "installment" in lc, f"{len(lc)} columns")

    _log("  running fresh-session inference test in a separate Python process")
    completed = _run_fresh_session_test()
    print(completed.stdout)
    if completed.returncode != 0 and completed.stderr:
        print(completed.stderr[-3000:])
    ctx["fresh_session_stdout"] = completed.stdout
    log.record("Fresh-session loading", "separate Python process loads preprocessor and transforms raw rows + new application",
               completed.returncode == 0, f"exit code {completed.returncode}")

    export.remove_staging()
    log.record("Backups untouched", "temporary staging folder removed",
               not config.STAGING_DIR.exists(), str(config.STAGING_DIR.relative_to(config.PROJECT_ROOT)))

    protected_after = export.snapshot_protected_paths()
    log.record("Backups untouched", "final_preprocessed_data/, processed_data/, step13_artifacts/ unchanged",
               protected_after == ctx["protected_before"],
               f"{len(protected_after)} files checked")

    ctx["complete"] = not log.failures
    export.save_json({
        "complete": ctx["complete"],
        "failures": log.failures,
        "warnings": log.warnings,
        "results": log.results,
    }, config.AUDIT_REPORT_PATH)
    _log(f"  {len(log.results)} checks: {len(log.failures)} FAIL, {len(log.warnings)} WARN")
    return ctx


def run_all():
    ctx = new_context()
    for stage in (stage_restore, stage_split, stage_feature_engineering, stage_fit,
                  stage_save_artifacts, stage_export, stage_final_audit):
        ctx = stage(ctx)
    from .report import write_reports
    write_reports(ctx)
    return ctx
