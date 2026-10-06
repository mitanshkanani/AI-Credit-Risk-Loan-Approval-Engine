"""Build deploy/app/samples.json and deploy/app/model_card.json.

    python deploy/build_samples.py

samples.json   six real VALIDATION applicants (two per decision), taken from their raw CSV rows, with only
               the fields the model uses. They give the web form complete, realistic starting points.
               The id is dropped; the actual outcome is kept so visitors can compare.
model_card.json  test metrics, policy and input fields shown on the page and at /api/model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.engine import CreditRiskEngine  # noqa: E402

APP = Path(__file__).resolve().parent / "app"
engine = CreditRiskEngine()
fields = ["issue_d", *engine.model_fields]

stage3 = pd.read_parquet(ROOT / "models" / "final" / "03_decision_layer" / "results" / "f3_validation_decisions.parquet")
candidates = pd.concat([stage3[stage3["decision"] == d].sample(400, random_state=11) for d in ("APPROVE", "REVIEW", "DECLINE")])
parts = []
for chunk in pd.read_csv(ROOT / "data" / "accepted_2007_to_2018Q4.csv", dtype=str, usecols=["id", *engine.raw_fields], chunksize=100_000):
    chunk["id"] = chunk["id"].str.strip()
    parts.append(chunk[chunk["id"].isin(candidates["id"])])
raw = pd.concat(parts).set_index("id")[fields]

samples, used_purposes = [], set()
for decision in ("APPROVE", "REVIEW", "DECLINE"):
    ids = candidates.loc[candidates["decision"] == decision, "id"]
    complete = [i for i in ids if raw.loc[i].isna().sum() <= 3 and isinstance(raw.loc[i, "emp_title"], str)]
    chosen = []
    for i in complete:   # vary the loan purpose across the six samples
        if raw.loc[i, "purpose"] not in used_purposes and len(chosen) < 2:
            chosen.append(i)
            used_purposes.add(raw.loc[i, "purpose"])
    for loan_id in chosen:
        record = {k: (None if pd.isna(v) else v.strip()) for k, v in raw.loc[loan_id].items()}
        for k, v in record.items():   # numbers as numbers (the API accepts strings too, but JSON is cleaner)
            try:
                record[k] = float(v) if v is not None and k not in ("term", "issue_d", "earliest_cr_line", "emp_length",
                                                                       "emp_title", "title", "purpose", "home_ownership",
                                                                       "verification_status", "addr_state") else v
            except ValueError:
                pass
        out = engine.score(record)
        target = int(candidates.set_index("id").loc[loan_id, "target"])
        samples.append({
            "name": f"{record['purpose'].replace('_', ' ').capitalize()} · ${record['loan_amnt']:,.0f} · {out['term_months']} months",
            "expected_decision": out["decision"],
            "actual_outcome": "defaulted" if target else "repaid",
            "application": record,
        })
        assert out["decision"] == decision
(APP / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")

summary = json.loads((ROOT / "models" / "final" / "05_final_test" / "results" / "f5_summary.json").read_text(encoding="utf-8"))
policy = engine.policy
card = {
    "model": engine.manifest["model_name"], "version": engine.manifest["version"],
    "trained_on": "LendingClub accepted loans issued 2012-08..2014-12 (388,124 loans)",
    "tested_on": f"sealed 2015 H2 test split ({summary['test_rows']:,} loans), opened once",
    "test": {"roc_auc": summary["metrics"]["roc_auc"], "ks": summary["metrics"]["ks"], "brier": summary["metrics"]["brier"],
             "mean_pd": summary["metrics"]["mean_pd"], "default_rate": summary["metrics"]["default_rate"],
             "lendingclub_interest_rate_roc_auc": summary["lendingclub_int_rate_roc_auc"]},
    "thresholds": policy["thresholds"], "risk_bands": policy["risk_bands"],
    "loss_model": {k: policy["loss_model"][k] for k in ("ead_share", "lgd", "formula")},
    "required_fields": ["loan_amnt", "term"], "input_fields": engine.model_fields,
    "known_limitations": ["60-month loans are under-predicted by about 2.5 percentage points on the test set",
                          "trained only on loans LendingClub approved; applicants it rejected were never seen",
                          "educational demo, not a lending decision"],
}
(APP / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
print(f"wrote {len(samples)} samples:", [(s["name"], s["expected_decision"], s["actual_outcome"]) for s in samples])
