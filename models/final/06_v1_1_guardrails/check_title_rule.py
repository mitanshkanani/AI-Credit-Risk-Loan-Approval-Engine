"""Effect of the v1.3 rule "a job / loan title not seen in training is treated as not provided" on the test set.

    python models/final/06_v1_1_guardrails/check_title_rule.py

Unlike the other guardrails, this rule changes a model INPUT (the target-encoded title value), so it can move PDs.
This reports how many test loans it touches and the test metrics with and without it. Nothing is tuned here.
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
from src.modeling import decision as dl  # noqa: E402
from src.modeling import evaluation as ev  # noqa: E402

engine = CreditRiskEngine()
X, y_frame = ev.load_split("test", feature_columns=engine.features, allow_test=True)
y = y_frame["target"].astype(int).to_numpy()
ids = set(y_frame["id"])
parts = [c[c["id"].str.strip().isin(ids)] for c in pd.read_csv(ROOT / "data" / "accepted_2007_to_2018Q4.csv", dtype=str,
                                                                usecols=["id", "emp_title", "title"], chunksize=200_000)]
raw = pd.concat(parts).assign(id=lambda d: d["id"].str.strip()).set_index("id").loc[y_frame["id"]]

te = engine.preprocessor.named_steps["encode"].named_transformers_["te"].named_steps["encode"]
names = list(engine.preprocessor.named_steps["encode"].named_transformers_["te"].feature_names_in_)
X_new = X.copy()
touched = np.zeros(len(X), bool)
result = {"test_loans": int(len(X))}
for field in ("emp_title", "title"):
    i = names.index(field)
    missing_value = te.encodings_[i][list(te.categories_[i]).index("__MISSING__")]
    norm = raw[field].str.strip().str.lower()
    unseen = (norm.notna() & ~norm.isin(engine.known_categories[field])).to_numpy()
    X_new.loc[unseen, field] = missing_value
    touched |= unseen
    result[f"{field}_unseen_loans"] = int(unseen.sum())

p_old = dl.calibrate(engine.booster.predict(X), engine.calibrator)
p_new = dl.calibrate(engine.booster.predict(X_new), engine.calibrator)
d_old, d_new = dl.decide(p_old, engine.policy), dl.decide(p_new, engine.policy)
result.update({
    "loans_touched": int(touched.sum()), "share_touched": float(touched.mean()),
    "decisions_changed": int((d_old != d_new).sum()),
    "roc_auc_v1_0": float(roc_auc_score(y, p_old)), "roc_auc_with_title_rule": float(roc_auc_score(y, p_new)),
    "mean_pd_v1_0": float(p_old.mean()), "mean_pd_with_title_rule": float(p_new.mean()), "default_rate": float(y.mean()),
    "approve_default_rate_v1_0": float(y[d_old == "APPROVE"].mean()),
    "approve_default_rate_with_title_rule": float(y[d_new == "APPROVE"].mean()),
    "approve_share_with_title_rule": float((d_new == "APPROVE").mean()),
})
out = Path(__file__).resolve().parent / "results" / "v1_3_title_rule_test_impact.json"
out.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
