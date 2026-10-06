"""Turn per-feature SHAP values into plain-language reason codes.

The model has 147 columns, but several describe one concept: one-hot dummies
(`addr_state_CA`, `purpose_*`, ...) and missing-value flags (`missingindicator_dti`).
SHAP values are additive, so a concept's contribution is the sum over its columns.
Each concept then gets a readable sentence built from the applicant's value.

Used by the Stage 2 analysis and, later, by the scoring API.
"""

import numpy as np
import pandas as pd

ONE_HOT = {
    "home_ownership_": ("home_ownership", "Home ownership"),
    "verification_status_": ("verification_status", "Income verification"),
    "purpose_": ("purpose", "Loan purpose"),
    "addr_state_": ("addr_state", "State"),
}

# concept -> (label, format). Formats: money, pct, count, months, years, term, years_from_months, rate, plain
CONCEPTS = {
    "term_months": ("Loan term", "term"),
    "loan_amnt": ("Loan amount", "money"),
    "annual_inc": ("Annual income", "money"),
    "dti": ("Debt-to-income ratio", "pct"),
    "fico_range_low": ("Credit score (FICO)", "plain"),
    "credit_history_months": ("Length of credit history", "years_from_months"),
    "emp_length_years": ("Employment length", "years"),
    "emp_title": ("Job title (historical default rate of similar titles)", "rate"),
    "zip_code": ("Area, 3-digit zip (historical default rate)", "rate"),
    "title": ("Loan title (historical default rate)", "rate"),
    "inq_last_6mths": ("Credit inquiries in the last 6 months", "count"),
    "mths_since_recent_inq": ("Months since the most recent credit inquiry", "months"),
    "acc_open_past_24mths": ("Accounts opened in the last 24 months", "count"),
    "num_tl_op_past_12m": ("Accounts opened in the last 12 months", "count"),
    "open_acc": ("Open credit lines", "count"),
    "total_acc": ("Total credit lines ever", "count"),
    "num_sats": ("Satisfactory accounts", "count"),
    "num_actv_bc_tl": ("Active bankcard accounts", "count"),
    "num_actv_rev_tl": ("Active revolving accounts", "count"),
    "num_bc_sats": ("Satisfactory bankcard accounts", "count"),
    "num_bc_tl": ("Bankcard accounts", "count"),
    "num_il_tl": ("Installment accounts", "count"),
    "num_op_rev_tl": ("Open revolving accounts", "count"),
    "num_rev_accts": ("Revolving accounts", "count"),
    "num_rev_tl_bal_gt_0": ("Revolving accounts with a balance", "count"),
    "mort_acc": ("Mortgage accounts", "count"),
    "revol_bal": ("Revolving balance", "money"),
    "revol_util": ("Revolving credit utilisation", "pct"),
    "bc_util": ("Bankcard utilisation", "pct"),
    "percent_bc_gt_75": ("Share of bankcards above 75% utilisation", "pct"),
    "bc_open_to_buy": ("Unused bankcard credit", "money"),
    "total_bc_limit": ("Total bankcard limit", "money"),
    "total_rev_hi_lim": ("Total revolving credit limit", "money"),
    "tot_hi_cred_lim": ("Total credit limit", "money"),
    "total_il_high_credit_limit": ("Total installment credit limit", "money"),
    "tot_cur_bal": ("Total current balance", "money"),
    "avg_cur_bal": ("Average balance per account", "money"),
    "total_bal_ex_mort": ("Total balance excluding mortgage", "money"),
    "mo_sin_old_il_acct": ("Age of oldest installment account", "months"),
    "mo_sin_old_rev_tl_op": ("Age of oldest revolving account", "months"),
    "mo_sin_rcnt_rev_tl_op": ("Months since newest revolving account", "months"),
    "mo_sin_rcnt_tl": ("Months since newest account", "months"),
    "mths_since_recent_bc": ("Months since newest bankcard", "months"),
    "delinq_2yrs": ("Delinquencies in the last 2 years", "count"),
    "mths_since_last_delinq": ("Months since last delinquency", "months"),
    "mths_since_recent_bc_dlq": ("Months since last bankcard delinquency", "months"),
    "mths_since_recent_revol_delinq": ("Months since last revolving delinquency", "months"),
    "mths_since_last_major_derog": ("Months since last major derogatory mark", "months"),
    "mths_since_last_record": ("Months since last public record", "months"),
    "num_accts_ever_120_pd": ("Accounts ever 120+ days past due", "count"),
    "num_tl_120dpd_2m": ("Accounts 120+ days past due (last 2 months)", "count"),
    "num_tl_30dpd": ("Accounts 30 days past due", "count"),
    "num_tl_90g_dpd_24m": ("Accounts 90+ days past due (last 24 months)", "count"),
    "pct_tl_nvr_dlq": ("Share of accounts never delinquent", "pct"),
    "acc_now_delinq": ("Accounts currently delinquent", "count"),
    "delinq_amnt": ("Amount currently past due", "money"),
    "pub_rec": ("Public records", "count"),
    "pub_rec_bankruptcies": ("Bankruptcies on record", "count"),
    "tax_liens": ("Tax liens", "count"),
    "collections_12_mths_ex_med": ("Collections in the last 12 months (non-medical)", "count"),
    "chargeoff_within_12_mths": ("Charge-offs in the last 12 months", "count"),
    "tot_coll_amt": ("Total amount ever in collections", "money"),
}
ONE_HOT_LABELS = {concept: label for concept, label in ONE_HOT.values()}
MISSING_PREFIX = "missingindicator_"


def concept_of(feature):
    """The concept a model column belongs to."""
    if feature.startswith(MISSING_PREFIX):
        return feature[len(MISSING_PREFIX):]
    for prefix, (concept, _) in ONE_HOT.items():
        if feature.startswith(prefix):
            return concept
    return feature


def label_of(concept):
    if concept in ONE_HOT_LABELS:
        return ONE_HOT_LABELS[concept]
    return CONCEPTS.get(concept, (concept.replace("_", " ").capitalize(), "plain"))[0]


def concept_shap(contrib, features):
    """(n, n_features[+1]) SHAP matrix -> DataFrame (n, n_concepts) summed per concept."""
    values = np.asarray(contrib)[:, :len(features)]
    frame = pd.DataFrame(values, columns=features)
    return frame.T.groupby([concept_of(f) for f in features], sort=False).sum().T


def _format(value, fmt):
    if fmt == "money":
        return f"${value:,.0f}"
    if fmt == "pct":
        return f"{value:.1f}%"
    if fmt == "count":
        return f"{value:.0f}"
    if fmt == "months":
        return f"{value:.0f} months"
    if fmt == "years":
        return f"{value:.0f} years" if value < 10 else "10+ years"
    if fmt == "years_from_months":
        return f"{value / 12:.1f} years"
    if fmt == "term":
        return f"{value:.0f} months"
    if fmt == "rate":
        return f"{value:.1%}"
    return f"{value:,.4g}"


def describe(concept, row):
    """Readable value of one concept for one applicant (row = Series of model inputs)."""
    if concept in ONE_HOT_LABELS:
        prefix = next(p for p, (c, _) in ONE_HOT.items() if c == concept)
        active = [k[len(prefix):] for k, v in row.items() if k.startswith(prefix) and v == 1]
        value = active[0] if active else "other"
        value = "other (rare category)" if value == "infrequent_sklearn" else value.replace("_", " ")
        return f"{label_of(concept)}: {value}"
    flag = MISSING_PREFIX + concept
    if flag in row.index and row[flag] == 1:
        return f"{label_of(concept)}: not reported"
    fmt = CONCEPTS.get(concept, (None, "plain"))[1]
    return f"{label_of(concept)}: {_format(row[concept], fmt)}"


def explain(concept_row, input_row, n_risk=4, n_protective=2):
    """Top concepts pushing risk up and down for one applicant.

    concept_row: Series of concept-level SHAP (log-odds); input_row: Series of model inputs.
    """
    ordered = concept_row.sort_values(ascending=False)
    risk = [{"concept": c, "reason": describe(c, input_row), "shap_log_odds": float(v)}
            for c, v in ordered.items() if v > 0][:n_risk]
    protective = [{"concept": c, "reason": describe(c, input_row), "shap_log_odds": float(v)}
                  for c, v in ordered[::-1].items() if v < 0][:n_protective]
    return {"risk_factors": risk, "protective_factors": protective}
