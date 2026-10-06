"""How many of the 212,801 test loans would the v1.1 guardrails change?

    python models/final/06_v1_1_guardrails/check_test_impact.py

The guardrail limits were set from the TRAINING data's ranges (not tuned on test). This only reports
their effect, so the Stage 5 test numbers can be quoted together with the v1.1 policy.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.engine import CreditRiskEngine  # noqa: E402
from src.engine import guardrails as gr  # noqa: E402
from src.modeling import decision as dl  # noqa: E402
from src.modeling import evaluation as ev  # noqa: E402

engine = CreditRiskEngine()
stage5 = ROOT / "models" / "final" / "05_final_test" / "results"
X, y_frame = ev.load_split("test", feature_columns=engine.features, allow_test=True)
y = y_frame["target"].astype(int).to_numpy()

# frozen v1.0 scoring of the test features (same as Stage 5)
pd_test = dl.calibrate(engine.booster.predict(X), engine.calibrator)
model_decision = dl.decide(pd_test, engine.policy)

# raw values needed for the scope rules and bureau coverage
ids = set(y_frame["id"])
cols = ["id", *gr.SCOPE.keys(), *engine.bureau_fields]
parts = [c[c["id"].str.strip().isin(ids)] for c in pd.read_csv(ROOT / "data" / "accepted_2007_to_2018Q4.csv", dtype=str,
                                                                usecols=list(dict.fromkeys(cols)), chunksize=100_000)]
raw = pd.concat(parts).assign(id=lambda d: d["id"].str.strip()).set_index("id").loc[y_frame["id"]]
num = raw[list(gr.SCOPE)].apply(pd.to_numeric, errors="coerce")
num["revol_util"] = raw["revol_util"].str.rstrip("%").pipe(pd.to_numeric, errors="coerce")
out_of_scope = np.zeros(len(raw), bool)
reasons = {}
for field, (low, high, _) in gr.SCOPE.items():
    v = num[field]
    bad = (v < low) | ((v > high) if high is not None else False)
    reasons[field] = int(bad.sum())
    out_of_scope |= bad.fillna(False).to_numpy()
short = (X["credit_history_months"] < gr.MIN_CREDIT_HISTORY_MONTHS).to_numpy()
reasons["credit_history_under_36_months"] = int(short.sum())
out_of_scope |= short
long_ = (X["credit_history_months"] > gr.MAX_CREDIT_HISTORY_MONTHS).to_numpy()
reasons["credit_history_over_70_years"] = int(long_.sum())
out_of_scope |= long_
lti = (num["loan_amnt"] / num["annual_inc"] > gr.MAX_LOAN_TO_INCOME).fillna(False).to_numpy()
reasons["loan_over_50pct_of_income"] = int(lti.sum())
out_of_scope |= lti
coverage = 1 - raw[engine.bureau_fields].isna().mean(axis=1).to_numpy()
thin = coverage < gr.MIN_BUREAU_COVERAGE

final = np.where(out_of_scope, "REFER", np.where((model_decision == "APPROVE") & thin, "REVIEW", model_decision))
changed = final != model_decision
approved = final == "APPROVE"
result = {
    "test_loans": int(len(y)),
    "changed_by_guardrails": int(changed.sum()),
    "changed_share": float(changed.mean()),
    "refer_out_of_scope": int((final == "REFER").sum()),
    "out_of_scope_by_rule": reasons,
    "approve_to_review_thin_bureau": int(((model_decision == "APPROVE") & thin & ~out_of_scope).sum()),
    "roc_auc_unchanged": float(roc_auc_score(y, pd_test)),
    "v1_0": {"approve_share": float((model_decision == "APPROVE").mean()), "approve_default_rate": float(y[model_decision == "APPROVE"].mean())},
    "guardrails_version": gr.GUARDRAILS_VERSION,
    "v1_1": {"approve_share": float(approved.mean()), "approve_default_rate": float(y[approved].mean()),
             "refer_share": float((final == "REFER").mean()),
             "refer_default_rate": float(y[final == "REFER"].mean()) if (final == "REFER").any() else None},
}
out = Path(__file__).resolve().parent / "results"
out.mkdir(exist_ok=True)
(out / "v1_1_test_impact.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
