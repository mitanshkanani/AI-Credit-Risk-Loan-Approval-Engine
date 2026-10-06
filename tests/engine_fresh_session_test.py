"""Fresh-session test of the complete scoring engine (Stage 4).

Run as its own Python process (no notebook state):

    python tests/engine_fresh_session_test.py

Raw application in -> final JSON out, checked against everything decided in Stages 1-3:
 1. the bundle loads and every file matches its manifest checksum; a tampered file is refused
 2. ORIGINAL raw CSV rows for 2,000 validation loans give exactly the PD, band, decision and
    expected loss saved by Stage 3 (validation only - the test split stays sealed)
 3. single-application scoring agrees with batch scoring
 4. a sparse hand-written application scores without NaN and lists its missing fields
 5. impossible inputs are rejected with a clear message
 6. same input -> same output; higher FICO never raises PD, higher DTI never lowers it
 7. latency of one application (warning only)

Writes models/final/04_scoring_package/results/f4_engine_test.json; exit code 0 only if all pass.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from src.engine import ApplicationError, CreditRiskEngine  # noqa: E402
from src.engine.scorer import BUNDLE_DIR  # noqa: E402

RAW_CSV = PROJECT_ROOT / "data" / "accepted_2007_to_2018Q4.csv"
STAGE3 = PROJECT_ROOT / "models" / "final" / "03_decision_layer" / "results" / "f3_validation_decisions.parquet"
OUT = PROJECT_ROOT / "models" / "final" / "04_scoring_package" / "results" / "f4_engine_test.json"
SAMPLE, SEED, LATENCY_TARGET_MS = 2000, 42, 200

SPARSE_APPLICATION = {   # only the basics a web form would ask for
    "loan_amnt": 12000, "term": "36 months", "annual_inc": 72000, "dti": 18.5, "fico_range_low": 700,
    "emp_length": "3 years", "home_ownership": "RENT", "purpose": "debt_consolidation", "addr_state": "CA",
    "earliest_cr_line": "Mar-2010", "revol_util": 41.2, "open_acc": 9, "total_acc": 20, "revol_bal": 8500,
    "inq_last_6mths": 1, "verification_status": "Verified", "emp_title": "Teacher", "title": "Debt consolidation",
}
INVALID = {
    "loan amount missing": {k: v for k, v in SPARSE_APPLICATION.items() if k != "loan_amnt"},
    "term of 48 months": {**SPARSE_APPLICATION, "term": 48},
    "FICO of 900": {**SPARSE_APPLICATION, "fico_range_low": 900},
    "negative income": {**SPARSE_APPLICATION, "annual_inc": -5},
    "bad credit-line date": {**SPARSE_APPLICATION, "earliest_cr_line": "2010-03-01"},
    "not a dict": ["loan_amnt", 1000],
    "core fields missing": {"loan_amnt": 5000, "term": 36},
    "unknown home ownership": {**SPARSE_APPLICATION, "home_ownership": "banana"},
    "credit line in the future": {**SPARSE_APPLICATION, "earliest_cr_line": "Jan-2030"},
    "negative delinquencies": {**SPARSE_APPLICATION, "delinq_2yrs": -3},
}
OUT_OF_SCOPE = {"FICO 300": {**SPARSE_APPLICATION, "fico_range_low": 300}, "DTI 60": {**SPARSE_APPLICATION, "dti": 60},
                "loan $1,000,000": {**SPARSE_APPLICATION, "loan_amnt": 1_000_000}}


def main():
    checks = []

    def check(name, passed, detail, required=True):
        status = "PASS" if passed else ("FAIL" if required else "WARNING")
        checks.append({"check": name, "status": status, "required": required, "detail": detail})
        print(f"{status:7s} {name} | {detail}", flush=True)

    # 1. load + integrity
    start = time.perf_counter()
    engine = CreditRiskEngine()
    check("engine loads in a fresh process; all bundle checksums match", True,
          f"{len(engine.manifest['sha256'])} files, {time.perf_counter() - start:.1f} s, scikit-learn {sklearn.__version__}")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bundle"
        shutil.copytree(BUNDLE_DIR, bad)
        calibrator = json.loads((bad / "calibrator.json").read_text(encoding="utf-8"))
        calibrator["b"] += 0.5
        (bad / "calibrator.json").write_text(json.dumps(calibrator), encoding="utf-8")
        try:
            CreditRiskEngine(bad)
            refused = False
        except RuntimeError as err:
            refused, message = True, str(err)
    check("a tampered bundle file is refused", refused, message if refused else "tampered calibrator was accepted")

    # 2. raw CSV rows -> Stage 3 decisions
    stage3 = pd.read_parquet(STAGE3)
    wanted = stage3.sample(SAMPLE, random_state=SEED).set_index("id")
    found = []
    for chunk in pd.read_csv(RAW_CSV, dtype=str, usecols=["id", *engine.raw_fields], chunksize=100_000):
        chunk["id"] = chunk["id"].str.strip()
        found.append(chunk[chunk["id"].isin(wanted.index)])
    raw = pd.concat(found).set_index("id").loc[wanted.index]
    scored = engine.score_batch(raw[engine.raw_fields].reset_index(drop=True))
    pd_diff = float(np.abs(scored["pd"].to_numpy() - wanted["pd"].to_numpy()).max())
    el_diff = float(np.abs(scored["expected_loss"].to_numpy() - wanted["expected_loss"].to_numpy()).max())
    same_band = bool((scored["risk_band"].to_numpy() == wanted["band"].to_numpy()).all())
    same_decision = bool((scored["decision"].to_numpy() == wanted["decision"].to_numpy()).all())
    check(f"raw CSV rows of {SAMPLE:,} validation loans reproduce Stage 3 exactly",
          pd_diff < 1e-9 and el_diff < 1e-6 and same_band and same_decision,
          f"max |PD diff| {pd_diff:.1e}, max |EL diff| ${el_diff:.1e}, bands equal {same_band}, decisions equal {same_decision}; "
          f"decisions: {scored['decision'].value_counts().to_dict()}")

    # 2b. cached target-encoder lookup == scikit-learn's own transform
    te = engine.preprocessor.named_steps["encode"].named_transformers_["te"].named_steps["encode"]
    fast_X = engine.transform(raw[engine.raw_fields].reset_index(drop=True))
    fast_fn, te.transform = te.transform, engine._sklearn_te_transform
    sklearn_X = engine.transform(raw[engine.raw_fields].reset_index(drop=True))
    te.transform = fast_fn
    unseen = {"emp_title": "zz never seen job", "title": None, "zip_code": "000xx"}
    fast_new = engine.transform([{**SPARSE_APPLICATION, **unseen, "issue_d": "Oct-2026"}])
    te.transform = engine._sklearn_te_transform
    sklearn_new = engine.transform([{**SPARSE_APPLICATION, **unseen, "issue_d": "Oct-2026"}])
    te.transform = fast_fn
    same = fast_X.equals(sklearn_X) and fast_new.equals(sklearn_new)
    check("cached target-encoder lookup is identical to scikit-learn (2,000 loans + unseen categories)", same,
          f"max |diff| {float(np.abs(fast_X.to_numpy() - sklearn_X.to_numpy()).max()):.1e}")

    # 3. single == batch
    singles = [engine.score(raw.iloc[i][engine.raw_fields].to_dict(), strict=False) for i in range(20)]
    single_vs_batch = max(abs(s["probability_of_default"] - scored["pd"].iloc[i]) for i, s in enumerate(singles))
    same_single_decisions = all(s["model_decision"] == scored["decision"].iloc[i] for i, s in enumerate(singles))
    check("single-application scoring agrees with batch scoring (20 loans)",
          single_vs_batch <= 5e-5 and same_single_decisions,
          f"max |diff| {single_vs_batch:.1e} (PD is rounded to 4 decimals in the JSON); decisions equal {same_single_decisions}")

    # 4. sparse hand-written application
    result = engine.score(SPARSE_APPLICATION)
    numbers = [result["probability_of_default"], result["expected_loss_usd"]] + [r["impact"] for r in result["reasons"]]
    check("sparse hand-written application -> complete JSON, no NaN",
          all(np.isfinite(numbers)) and result["decision"] in ("APPROVE", "REVIEW", "DECLINE") and len(result["reasons"]) > 0,
          f"PD {result['probability_of_default']}, {result['risk_band']}, {result['decision']}, EL ${result['expected_loss_usd']}, "
          f"{len(result['missing_fields'])} missing fields listed")

    # 5. invalid inputs
    rejected = {}
    for name, app in INVALID.items():
        try:
            engine.score(app)
            rejected[name] = "ACCEPTED"
        except ApplicationError as err:
            rejected[name] = str(err)
    check("impossible inputs are rejected with a message", all(v != "ACCEPTED" for v in rejected.values()),
          "; ".join(f"{k}: {v}" for k, v in rejected.items()))

    # 5b. v1.1 guardrails
    refer = {name: engine.score(app)["decision"] for name, app in OUT_OF_SCOPE.items()}
    thin = engine.score(SPARSE_APPLICATION)
    check("v1.1: out-of-scope applicants are REFERred, thin bureau data never auto-approved",
          all(d == "REFER" for d in refer.values()) and thin["decision"] != "APPROVE",
          f"{refer}; sparse application: model {thin['model_decision']} -> {thin['decision']} "
          f"(bureau coverage {thin['bureau_data_coverage']:.0%})")

    # 6. determinism and monotonic behaviour
    again = engine.score(SPARSE_APPLICATION)
    check("same input -> same output",
          {k: v for k, v in again.items() if k != "latency_ms"} == {k: v for k, v in result.items() if k != "latency_ms"}, "")
    fico = [engine.score({**SPARSE_APPLICATION, "fico_range_low": f})["probability_of_default"] for f in range(660, 851, 15)]
    dti = [engine.score({**SPARSE_APPLICATION, "dti": d})["probability_of_default"] for d in range(0, 41, 4)]
    check("higher FICO never raises PD; higher DTI never lowers it",
          bool(np.all(np.diff(fico) <= 0) and np.all(np.diff(dti) >= 0)),
          f"FICO 660->850: PD {fico[0]:.3f}->{fico[-1]:.3f}; DTI 0->40: PD {dti[0]:.3f}->{dti[-1]:.3f}")

    # 7. latency
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        engine.score(SPARSE_APPLICATION)
        times.append((time.perf_counter() - t0) * 1000)
    median_ms = float(np.median(times))
    check(f"one application scored in under {LATENCY_TARGET_MS} ms (median of 50)", median_ms < LATENCY_TARGET_MS,
          f"median {median_ms:.0f} ms, p95 {np.percentile(times, 95):.0f} ms", required=False)

    passed = all(c["status"] == "PASS" for c in checks if c["required"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"passed": passed, "checks": checks, "example_output": result,
                               "fico_sweep": fico, "dti_sweep": dti, "latency_ms_median": median_ms}, indent=2), encoding="utf-8")
    print("ENGINE FRESH-SESSION TEST:", "PASSED" if passed else "FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
