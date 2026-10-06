"""v1.1 policy guardrails around the frozen v1.0 model (added after an independent audit).

The model, calibrator and thresholds are unchanged. These rules only decide WHEN the model may be
trusted for an application:

1. Strict input checks: core fields required, categories must be ones seen in training,
   counts and amounts cannot be negative, dates cannot be in the future.
2. Scope: LendingClub only funded applicants inside certain limits (FICO >= 660, DTI < 40, ...),
   so the model never saw anyone outside them. Such applications are REFERred to a person
   instead of being scored as if they were normal.
3. Data coverage: if most credit-bureau fields are missing, the PD rests on imputed values, so an
   APPROVE is downgraded to REVIEW.
4. Loan title: LendingClub's form filled `title` from the purpose; a blank title fell into a
   rare "no title" group with an artificially low default rate. A missing title is now filled
   from the purpose, as LendingClub did.
"""

from datetime import date

import pandas as pd

GUARDRAILS_VERSION = "1.3.0"

# Required for a hand-entered application (besides loan_amnt and term, checked by the engine)
CORE_FIELDS = ["annual_inc", "dti", "fico_range_low", "earliest_cr_line", "home_ownership",
               "verification_status", "purpose", "addr_state", "revol_util", "inq_last_6mths"]

CATEGORIES = {
    "home_ownership": {"MORTGAGE", "OWN", "RENT"},
    "verification_status": {"Not Verified", "Source Verified", "Verified"},
    "purpose": {"car", "credit_card", "debt_consolidation", "home_improvement", "house", "major_purchase", "medical",
                "moving", "other", "renewable_energy", "small_business", "vacation", "wedding"},
    "addr_state": {"AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS",
                   "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "NE", "NH", "NJ", "NM", "NV", "NY",
                   "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY"},
    "emp_length": {"< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years", "6 years", "7 years", "8 years",
                   "9 years", "10+ years"},
}

# Training data (388,124 loans, 2012-08..2014-12) covers only these ranges
SCOPE = {
    "fico_range_low": (660, 850, "FICO score below 660: LendingClub did not lend below 660, so the model has never seen such applicants"),
    "dti": (0, 40, "debt-to-income above 40%: outside LendingClub's lending limit and the model's training data"),
    "loan_amnt": (1000, 35000, "loan amount outside $1,000-$35,000: the range LendingClub lent in 2012-2014"),
    "annual_inc": (3000, None, "annual income below $3,000: outside the training data"),
    "revol_util": (0, 150, "revolving utilisation above 150%: outside the training data"),
    "inq_last_6mths": (0, 8, "more than 8 credit inquiries in 6 months: outside the training data"),
}
MIN_CREDIT_HISTORY_MONTHS = 36
MAX_CREDIT_HISTORY_MONTHS = 842          # longest history in the training data (~70 years)
MAX_LOAN_TO_INCOME = 0.5                 # LendingClub never lent more than 50% of annual income (training max = 0.50)

# fields that must be whole numbers (counts / months)
def is_count_field(field):
    return field.startswith(("num_", "mths_", "mo_sin_", "open_", "inq_", "acc_", "total_acc", "delinq_2yrs", "pub_rec",
                             "tax_liens", "chargeoff", "collections", "mort_acc"))
MIN_BUREAU_COVERAGE = 0.5

TITLE_FROM_PURPOSE = {
    "debt_consolidation": "Debt consolidation", "credit_card": "Credit card refinancing",
    "home_improvement": "Home improvement", "major_purchase": "Major purchase", "small_business": "Business",
    "car": "Car financing", "medical": "Medical expenses", "moving": "Moving and relocation", "vacation": "Vacation",
    "house": "Home buying", "other": "Other", "renewable_energy": "Green loan", "wedding": "Wedding Loan",
}


class InputError(ValueError):
    pass


def _missing(value):
    return value is None or (isinstance(value, str) and value.strip() == "") or (isinstance(value, float) and value != value)


def validate(app, numeric_fields, require_core=True):
    """Strict checks on one raw application (dict). Raises InputError with a readable message."""
    if require_core:
        absent = [f for f in CORE_FIELDS if _missing(app.get(f))]
        if absent:
            raise InputError(f"required fields missing: {', '.join(absent)}")
    for field, allowed in CATEGORIES.items():
        value = app.get(field)
        if not _missing(value) and str(value).strip() not in allowed:
            raise InputError(f"{field} must be one of: {', '.join(sorted(allowed))}")
    for field in numeric_fields:
        value = app.get(field)
        if _missing(value):
            continue
        number = to_number(field, value)
        if number < 0:
            raise InputError(f"{field} cannot be negative")
        if is_count_field(field) and number != int(number):
            raise InputError(f"{field} must be a whole number")
    if not _missing(app.get("open_acc")) and not _missing(app.get("total_acc")):
        if to_number("open_acc", app["open_acc"]) > to_number("total_acc", app["total_acc"]):
            raise InputError("open_acc (open credit lines) cannot exceed total_acc (total credit lines)")
    for field in ("earliest_cr_line", "issue_d"):
        if not _missing(app.get(field)):
            when = pd.to_datetime(str(app[field]).strip(), format="%b-%Y", errors="coerce")
            if pd.isna(when):
                raise InputError(f"{field} must look like 'Mar-2010'")
            if when > pd.Timestamp(date.today()):
                raise InputError(f"{field} cannot be in the future")


def to_number(field, value):
    """Strict number parsing: rejects booleans, lists, NaN/inf and text, with a readable message."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise InputError(f"{field} must be a number")
    try:
        number = float(str(value).strip().rstrip("%")) if isinstance(value, str) else float(value)
    except ValueError:
        raise InputError(f"{field} must be a number") from None
    if number != number or number in (float("inf"), float("-inf")):
        raise InputError(f"{field} must be a finite number")
    return number


def fill_title(app):
    """Fill a blank loan title from the purpose, as LendingClub's application form did."""
    if _missing(app.get("title")) and not _missing(app.get("purpose")):
        app["title"] = TITLE_FROM_PURPOSE.get(str(app["purpose"]).strip(), "Other")
        return True
    return False


def scope_issues(app, credit_history_months):
    issues = []
    for field, (low, high, message) in SCOPE.items():
        value = app.get(field)
        if _missing(value):
            continue
        value = float(value)
        if value < low or (high is not None and value > high):
            issues.append(message)
    if credit_history_months is not None and credit_history_months == credit_history_months:
        if credit_history_months < MIN_CREDIT_HISTORY_MONTHS:
            issues.append("credit history shorter than 3 years: LendingClub required at least 36 months")
        if credit_history_months > MAX_CREDIT_HISTORY_MONTHS:
            issues.append("credit history longer than 70 years: implausible and outside the training data")
    loan, income = app.get("loan_amnt"), app.get("annual_inc")
    if not _missing(loan) and not _missing(income) and float(income) > 0 and float(loan) / float(income) > MAX_LOAN_TO_INCOME:
        issues.append(f"loan is {float(loan) / float(income):.0%} of annual income: LendingClub never lent more than 50% of income")
    return issues


def final_decision(model_decision, issues, bureau_coverage):
    """Combine the frozen model decision with the guardrails -> (decision, notes)."""
    if issues:
        return "REFER", ["Out of scope, sent to a credit officer (the model's score is not valid here): " + "; ".join(issues)]
    if model_decision == "APPROVE" and bureau_coverage < MIN_BUREAU_COVERAGE:
        return "REVIEW", [f"Only {bureau_coverage:.0%} of credit-bureau fields were provided: the PD relies on typical "
                          "values, so the application goes to manual review instead of automatic approval"]
    return model_decision, []


DEROGATORY = {"delinq_2yrs", "pub_rec", "pub_rec_bankruptcies", "tax_liens", "chargeoff_within_12_mths",
              "collections_12_mths_ex_med", "acc_now_delinq", "delinq_amnt", "tot_coll_amt", "num_accts_ever_120_pd",
              "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m"}


def is_adverse(concept, model_row):
    """True if this concept describes something bad on the applicant's file (so it must not be shown as a strength)."""
    if concept in DEROGATORY:
        return float(model_row.get(concept, 0) or 0) > 0
    if concept == "pct_tl_nvr_dlq":
        return float(model_row.get(concept, 100)) < 100
    if concept in ("mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
                   "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq"):
        return model_row.get("missingindicator_" + concept, 1) == 0     # a recorded delinquency exists
    return False
