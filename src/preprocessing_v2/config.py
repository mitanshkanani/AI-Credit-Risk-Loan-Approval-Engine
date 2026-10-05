"""Central configuration for accepted-loan preprocessing V2.

Every path, date boundary, column group and threshold used by the V2
pipeline lives here so that each decision is documented in one place.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ACCEPTED_PATH = PROJECT_ROOT / "data" / "accepted_2007_to_2018Q4.csv"

# Step 10 output of accepted_preprocessing.ipynb (Steps 1-10 are kept).
BASE_DATASET_PATH = PROJECT_ROOT / "accepted_quality_clean.csv"

OUTPUT_DIR = PROJECT_ROOT / "final_preprocessed_data_v2"
# Temporary per-split raw rows (keeps memory low); deleted after the run.
STAGING_DIR = OUTPUT_DIR / "_staging"
TRANSFORM_CHUNK_ROWS = 100_000
ARTIFACT_DIR = PROJECT_ROOT / "artifacts_v2"

PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor_v2.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata_v2.json"
AUDIT_REPORT_PATH = ARTIFACT_DIR / "audit_report_v2.json"
SUMMARY_REPORT_PATH = ARTIFACT_DIR / "preprocessing_v2_report.md"
FEATURE_LIST_PATH = ARTIFACT_DIR / "feature_list_v2.csv"
REMOVED_FEATURES_PATH = ARTIFACT_DIR / "removed_features_v2.csv"
ENGINEERED_FEATURES_PATH = ARTIFACT_DIR / "engineered_features_v2.csv"
SINGLE_FEATURE_AUC_PATH = ARTIFACT_DIR / "single_feature_auc_v2.csv"
FRESH_SESSION_RESULT_PATH = ARTIFACT_DIR / "fresh_session_test_v2.json"

# V1 outputs: treated as a read-only backup. V2 must never modify them.
PROTECTED_PATHS = [
    PROJECT_ROOT / "final_preprocessed_data",
    PROJECT_ROOT / "processed_data",
    PROJECT_ROOT / "step13_artifacts",
]

# ------------------------------------------------------------------
# Target (identical to Step 3 of accepted_preprocessing.ipynb)
# ------------------------------------------------------------------
TARGET_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# ------------------------------------------------------------------
# Temporal split (issue month, start inclusive / end exclusive)
# ------------------------------------------------------------------
SPLIT_WINDOWS = {
    "train": ("2012-08-01", "2015-01-01"),
    "validation": ("2015-01-01", "2015-07-01"),
    "test": ("2015-07-01", "2016-01-01"),
    # Optional stress-test population only: these cohorts are largely
    # unresolved at the 2018Q4 snapshot, so their default rate is biased.
    "stress_2016_2018": ("2016-01-01", "2019-01-01"),
}
MODELING_SPLITS = ["train", "validation", "test"]
STRESS_SPLIT = "stress_2016_2018"
# Loans issued before Aug-2012 predate the credit-bureau fields
# (tot_cur_bal, num_*, mo_sin_*, ...) and are excluded entirely.
EXCLUDED_LABEL = "excluded_pre_2012_08"

# ------------------------------------------------------------------
# Restoration safety check: columns compared row-by-row between the
# raw CSV (eligible rows, original order) and the Step 10 dataset.
# ------------------------------------------------------------------
ALIGNMENT_NUMERIC_COLUMNS = ["loan_amnt", "int_rate", "annual_inc", "dti", "revol_util", "fico_range_low"]
ALIGNMENT_TEXT_COLUMNS = ["term", "grade", "emp_title", "title", "zip_code", "earliest_cr_line"]

KEY_COLUMNS = ["id", "issue_d"]
TARGET_COLUMN = "target"

# ------------------------------------------------------------------
# Column groups used by feature engineering and the preprocessor
# ------------------------------------------------------------------
# The application/issue month. At inference time this is the month the
# application is scored. It is only used to derive credit history length.
APPLICATION_DATE_COLUMN = "issue_d"

ONE_HOT_COLUMNS = [
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
    "verification_status_joint",
]
TARGET_ENCODED_COLUMNS = ["zip_code", "emp_title", "title"]
CATEGORICAL_COLUMNS = ONE_HOT_COLUMNS + TARGET_ENCODED_COLUMNS

# Loaded as compact pyarrow strings (memory only; values are unchanged).
TEXT_INPUT_COLUMNS = CATEGORICAL_COLUMNS + [
    "term", "emp_length", "earliest_cr_line", "sec_app_earliest_cr_line",
]

# Raw columns replaced by engineered numeric features.
REPLACED_BY_ENGINEERING = {
    "earliest_cr_line": "credit_history_months",
    "sec_app_earliest_cr_line": "sec_app_credit_history_months",
    "term": "term_months",
    "emp_length": "emp_length_years",
}

# Removed by design before fitting (documented decisions).
DROPPED_BY_DESIGN = {
    "desc": "Free-text loan description; Step 9 marked it DROP_INITIAL_VERSION (needs a separate NLP pipeline).",
}

# LendingClub-derived risk variables kept for the A/B modeling comparison.
LC_RISK_SOURCE_COLUMNS = ["grade", "sub_grade", "int_rate", "installment"]

# Columns that must never appear among model features (Step 6/7 lists).
FORBIDDEN_FEATURE_SOURCES = [
    "loan_status", "target", "id", "member_id", "url", "policy_code",
    "initial_list_status", "issue_d",
    "funded_amnt", "funded_amnt_inv", "pymnt_plan", "out_prncp",
    "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d", "last_fico_range_high",
    "last_fico_range_low", "hardship_flag", "hardship_type",
    "hardship_reason", "hardship_status", "deferral_term",
    "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "disbursement_method", "debt_settlement_flag",
    "debt_settlement_flag_date", "settlement_status", "settlement_date",
    "settlement_amount", "settlement_percentage", "settlement_term",
]

# ------------------------------------------------------------------
# Fitted-preprocessing parameters
# ------------------------------------------------------------------
MAX_TRAIN_MISSING_RATE = 0.95      # drop columns missing in >95% of TRAIN rows
ONE_HOT_MIN_FREQUENCY = 500        # rarer train categories -> "infrequent"
MISSING_CATEGORY = "__MISSING__"
TARGET_ENCODER_CV = 5
RANDOM_STATE = 42

# ------------------------------------------------------------------
# Audit parameters
# ------------------------------------------------------------------
SINGLE_FEATURE_AUC_REVIEW_THRESHOLD = 0.80   # WARNING / review flag only
TARGET_SPOT_CHECK_SIZE = 1000
FRESH_SESSION_SAMPLE_SIZE = 5
