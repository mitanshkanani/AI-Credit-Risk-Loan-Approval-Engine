"""Decision layer: calibrated PD -> risk band, decision and expected loss.

Every number used here lives in one plain JSON policy file
(models/final/03_decision_layer/results/decision_policy.json), written by Stage 3.
This module only applies it, so the same rules run in the notebook, the scoring package
and the live demo.
"""

import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = PROJECT_ROOT / "models" / "final" / "03_decision_layer" / "results" / "decision_policy.json"


def load_policy(path=POLICY_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def calibrate(pd_raw, calibrator):
    """Platt scaling: PD_cal = 1 / (1 + exp(-(a * logit(PD_raw) + b)))."""
    p = np.clip(np.asarray(pd_raw, dtype=float), 1e-6, 1 - 1e-6)
    z = calibrator["a"] * np.log(p / (1 - p)) + calibrator["b"]
    return 1 / (1 + np.exp(-z))


def risk_band(pd_cal, policy):
    """Band name for each PD; bands are [lower, upper) intervals of calibrated PD."""
    bands = policy["risk_bands"]
    upper = np.array([b["pd_upper"] for b in bands[:-1]])
    idx = np.searchsorted(upper, np.asarray(pd_cal, dtype=float), side="right")
    return np.array([bands[i]["band"] for i in idx])


def decide(pd_cal, policy):
    """APPROVE if PD <= approve threshold, DECLINE if PD >= decline threshold, otherwise REVIEW."""
    p = np.asarray(pd_cal, dtype=float)
    t = policy["thresholds"]
    return np.where(p <= t["approve_max_pd"], "APPROVE", np.where(p >= t["decline_min_pd"], "DECLINE", "REVIEW"))


def expected_loss(pd_cal, loan_amnt, term_months, policy):
    """EL = PD x EAD x LGD, with EAD = loan amount x (share still owed at default, by term)."""
    loss = policy["loss_model"]
    term = np.asarray(term_months).astype(int)
    ead_share = np.where(term == 60, loss["ead_share"]["60"], loss["ead_share"]["36"])
    lgd = np.where(term == 60, loss["lgd"]["60"], loss["lgd"]["36"])
    ead = np.asarray(loan_amnt, dtype=float) * ead_share
    return np.asarray(pd_cal, dtype=float) * ead * lgd


def apply_policy(pd_cal, loan_amnt, term_months, policy):
    """Vectorised decision for many applicants: dict of arrays."""
    el = expected_loss(pd_cal, loan_amnt, term_months, policy)
    return {
        "pd": np.asarray(pd_cal, dtype=float),
        "risk_band": risk_band(pd_cal, policy),
        "decision": decide(pd_cal, policy),
        "expected_loss": el,
        "expected_loss_rate": el / np.asarray(loan_amnt, dtype=float),
    }
