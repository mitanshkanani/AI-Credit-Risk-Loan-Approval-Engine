"""Stages 5-6 - Artifact saving and dataset export."""

import json
import platform

import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn

from . import config


def snapshot_protected_paths():
    """(relative path, size, mtime) for every file in the V1 backup folders."""
    snapshot = {}
    for root in config.PROTECTED_PATHS:
        if not root.exists():
            snapshot[str(root.name)] = "MISSING"
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            stat = path.stat()
            snapshot[str(path.relative_to(config.PROJECT_ROOT))] = [stat.st_size, stat.st_mtime_ns]
    return snapshot


def prepare_output_dirs():
    for directory in (config.OUTPUT_DIR, config.ARTIFACT_DIR):
        if directory.resolve() in {p.resolve() for p in config.PROTECTED_PATHS}:
            raise RuntimeError(f"Refusing to write into protected path {directory}")
        directory.mkdir(parents=True, exist_ok=True)


def dataset_paths(split):
    return (
        config.OUTPUT_DIR / f"X_{split}.parquet",
        config.OUTPUT_DIR / f"y_{split}.parquet",
    )


def export_split(split, X, y_frame):
    """X = features only; y = id, issue_d, target in the same row order."""
    x_path, y_path = dataset_paths(split)
    X.reset_index(drop=True).to_parquet(x_path, index=False)
    y_frame.reset_index(drop=True).to_parquet(y_path, index=False)
    return x_path, y_path


def export_split_in_chunks(split, X_chunks, y_frame, on_chunk=None):
    """Stream transformed chunks into one Parquet file (constant memory)."""
    x_path, y_path = dataset_paths(split)
    writer = None
    try:
        for X in X_chunks:
            if on_chunk is not None:
                on_chunk(X)
            table = pa.Table.from_pandas(X.reset_index(drop=True), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(x_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    y_frame.reset_index(drop=True).to_parquet(y_path, index=False)
    return x_path, y_path


def stage_raw_split(split, frame):
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    frame.reset_index(drop=True).to_parquet(config.STAGING_DIR / f"raw_{split}.parquet", index=False)


def load_raw_split(split):
    return pd.read_parquet(config.STAGING_DIR / f"raw_{split}.parquet")


def remove_staging():
    if config.STAGING_DIR.exists():
        for path in config.STAGING_DIR.glob("raw_*.parquet"):
            path.unlink()
        config.STAGING_DIR.rmdir()


def save_preprocessor(preprocessor):
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    return config.PREPROCESSOR_PATH


def save_json(payload, path):
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def environment_info():
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }
