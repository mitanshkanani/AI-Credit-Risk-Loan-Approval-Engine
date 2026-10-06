"""Tests for the live-demo API (Stage 6).

    python -m pytest tests/test_api.py -q        (or: python tests/test_api.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from deploy.app.main import app  # noqa: E402


def client():
    return TestClient(app)


def test_health_and_page():
    with client() as c:
        assert c.get("/api/health").json() == {"status": "ok", "model": "M3-B-mono-nozip", "version": "1.0.0"}
        page = c.get("/")
        assert page.status_code == 200 and "Credit Risk" in page.text
        assert c.get("/docs").status_code == 200


def test_samples_score_as_labelled():
    with client() as c:
        samples = c.get("/api/samples").json()
        assert len(samples) == 6
        for s in samples:
            r = c.post("/api/score", json=s["application"])
            assert r.status_code == 200, r.text
            assert r.json()["decision"] == s["expected_decision"]


def test_minimal_application_needs_core_fields():
    with client() as c:
        r = c.post("/api/score", json={"loan_amnt": 10000, "term": 36})
        assert r.status_code == 422 and "required fields missing" in r.json()["detail"]


def test_guardrails():
    with client() as c:
        s = c.get("/api/samples").json()[0]["application"]
        assert c.post("/api/score", json={**s, "fico_range_low": 300}).json()["decision"] == "REFER"
        core = {k: s[k] for k in ("loan_amnt", "term", "annual_inc", "dti", "fico_range_low", "earliest_cr_line",
                                  "home_ownership", "verification_status", "purpose", "addr_state", "revol_util", "inq_last_6mths")}
        body = c.post("/api/score", json=core).json()
        assert body["decision"] != "APPROVE" and body["bureau_data_coverage"] < 0.5


def test_bad_inputs_get_422_with_a_message():
    with client() as c:
        for bad, words in [({"term": 36}, "loan_amnt"), ({"loan_amnt": 5000, "term": 48}, "36 or 60"),
                           ({"loan_amnt": 5000, "term": 36, "fico_range_low": 950}, "fico"),
                           ({"loan_amnt": -1, "term": 36}, "loan_amnt"),
                           ({"loan_amnt": 5000, "term": 36}, "required fields")]:
            r = c.post("/api/score", json=bad)
            assert r.status_code == 422, bad
            assert words in str(r.json()["detail"]), r.json()


def test_model_card():
    with client() as c:
        card = c.get("/api/model").json()
        assert card["test"]["roc_auc"] > card["test"]["lendingclub_interest_rate_roc_auc"]
        assert len(card["input_fields"]) == 65


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
