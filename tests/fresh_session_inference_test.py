"""Fresh-session inference test for the saved V2 preprocessor.

Run as its own Python process (no notebook state):

    python tests/fresh_session_inference_test.py

1. Loads artifacts_v2/preprocessor_v2.joblib with joblib only.
2. Pulls the ORIGINAL raw CSV rows for 5 test-split loan ids, transforms
   them, and requires the result to equal the exported X_test rows.
3. Transforms a hand-written new application (unseen zip code, missing
   emp_title, missing dti) and checks the fallbacks are the training ones.

Writes artifacts_v2/fresh_session_test_v2.json; exit code 0 only if all pass.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.preprocessing_v2 import config, inference  # noqa: E402

NEW_APPLICATION = {
    "issue_d": "Oct-2026",              # application month
    "loan_amnt": 12000,
    "term": " 36 months",
    "int_rate": 11.5,
    "installment": 395.7,
    "grade": "B",
    "sub_grade": "B4",
    "emp_title": None,                   # missing -> learned __MISSING__ encoding
    "emp_length": "3 years",
    "home_ownership": "RENT",
    "annual_inc": 72000,
    "verification_status": "Verified",
    "purpose": "debt_consolidation",
    "title": "Debt consolidation",
    "zip_code": "000xx",                 # no US zip prefix 000 -> never seen in training
    "addr_state": "CA",
    "dti": None,                         # missing -> training median
    "earliest_cr_line": "Mar-2010",
    "fico_range_low": 700,
    "fico_range_high": 704,
    "open_acc": 9,
    "revol_bal": 8500,
    "revol_util": 41.2,
    "total_acc": 20,
    "application_type": "Individual",
}


def main():
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    preprocessor = inference.load_preprocessor()
    metadata = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    feature_names = metadata["feature_names"]
    input_columns = inference.required_input_columns(preprocessor)
    check("preprocessor loads in a fresh process", True, str(config.PREPROCESSOR_PATH))

    # ---------------- 1. Real raw rows -> must equal exported X_test rows
    y_test = pd.read_parquet(config.OUTPUT_DIR / "y_test.parquet")
    positions = np.sort(
        np.random.default_rng(config.RANDOM_STATE).choice(
            len(y_test), size=config.FRESH_SESSION_SAMPLE_SIZE, replace=False)
    )
    wanted_ids = y_test["id"].iloc[positions].tolist()

    found = {}
    for chunk in pd.read_csv(config.RAW_ACCEPTED_PATH, dtype=str, chunksize=250_000):
        hit = chunk[chunk["id"].str.strip().isin(wanted_ids)]
        for _, row in hit.iterrows():
            found[row["id"].strip()] = row
        if len(found) == len(wanted_ids):
            break
    raw_rows = pd.DataFrame([found[i] for i in wanted_ids])[input_columns]

    transformed = preprocessor.transform(raw_rows)
    saved = pd.read_parquet(config.OUTPUT_DIR / "X_test.parquet").iloc[positions]
    columns_match = list(transformed.columns) == list(saved.columns) == feature_names
    max_abs_diff = float(np.nanmax(np.abs(transformed.to_numpy() - saved.to_numpy())))
    values_match = np.allclose(transformed.to_numpy(), saved.to_numpy(), rtol=1e-9, atol=1e-9)
    check("raw CSV rows for 5 test ids reproduce exported X_test rows exactly",
          columns_match and values_match,
          f"ids = {wanted_ids}; max |diff| = {max_abs_diff:.3e}; columns match = {columns_match}")

    # ---------------- 2. Hand-written new application
    out = inference.transform_applications([NEW_APPLICATION], preprocessor)
    values = out.to_numpy()
    check("new application -> 1 row with the full feature set",
          out.shape == (1, len(feature_names)) and list(out.columns) == feature_names,
          f"shape = {out.shape}")
    check("new application -> no NaN / inf",
          not np.isnan(values).any() and not np.isinf(values).any(),
          f"NaN = {int(np.isnan(values).sum())}, inf = {int(np.isinf(values).sum())}")

    encoder = preprocessor.named_steps["encode"]
    te_pipe = encoder.named_transformers_["te"]
    te = te_pipe.named_steps["encode"]
    te_columns = list(te_pipe.feature_names_in_)

    zip_categories = set(te.categories_[te_columns.index("zip_code")])
    zip_unseen = NEW_APPLICATION["zip_code"] not in zip_categories
    zip_value = float(out["zip_code"].iloc[0])
    check("unseen zip_code falls back to the TRAIN prior default rate",
          zip_unseen and np.isclose(zip_value, te.target_mean_),
          f"'{NEW_APPLICATION['zip_code']}' absent from {len(zip_categories)} train zips = {zip_unseen}; "
          f"encoded {zip_value:.6f} vs train prior {te.target_mean_:.6f}")

    emp_idx = te_columns.index("emp_title")
    categories = list(te.categories_[emp_idx])
    if config.MISSING_CATEGORY in categories:
        expected_missing = te.encodings_[emp_idx][categories.index(config.MISSING_CATEGORY)]
    else:
        expected_missing = te.target_mean_
    emp_value = float(out["emp_title"].iloc[0])
    check("missing emp_title uses the TRAIN-learned __MISSING__ encoding",
          np.isclose(emp_value, expected_missing),
          f"encoded {emp_value:.6f} vs expected {expected_missing:.6f}")

    num_pipe = encoder.named_transformers_["num"]
    num_columns = list(num_pipe.feature_names_in_)
    dti_median = num_pipe.named_steps["impute"].statistics_[num_columns.index("dti")]
    dti_value = float(out["dti"].iloc[0])
    dti_flag = float(out["missingindicator_dti"].iloc[0]) if "missingindicator_dti" in out else None
    check("missing dti is imputed with the TRAIN median",
          np.isclose(dti_value, dti_median),
          f"dti {dti_value} vs train median {dti_median}; missing flag = {dti_flag}")

    history = float(out["credit_history_months"].iloc[0])
    check("credit_history_months computed from application month",
          history == (2026 - 2010) * 12 + (10 - 3),
          f"Mar-2010 -> Oct-2026 = {history} months")

    passed = all(c["status"] == "PASS" for c in checks)
    result = {"passed": passed, "checks": checks}
    config.FRESH_SESSION_RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for c in checks:
        print(f"{c['status']}  {c['check']}  |  {c['detail']}")
    print("FRESH-SESSION TEST:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
