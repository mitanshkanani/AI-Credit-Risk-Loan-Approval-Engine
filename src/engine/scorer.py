"""The credit-risk engine: one raw loan application in, one decision JSON out.

    raw application (dict)
      -> preprocessor_v2.joblib         (feature engineering, imputation, encoding; fitted on TRAIN)
      -> final_model.txt                (LightGBM M3-B-mono-nozip, 146 features)
      -> calibrator.json                (Platt scaling, fitted on validation)
      -> decision_policy.json           (risk band, approve/review/decline, expected loss)
      -> reason codes                   (SHAP -> plain-language reasons)

Everything is read from one bundle folder whose manifest.json records the checksums of the
files and the library versions they were built with. The engine refuses to start if a file
was changed or if scikit-learn differs from the version that fitted the preprocessor.

    from src.engine import CreditRiskEngine
    engine = CreditRiskEngine()
    engine.score({"loan_amnt": 12000, "term": 36, "annual_inc": 72000, "fico_range_low": 700, ...})
"""

import hashlib
import json
import time
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from src.modeling import decision as dl
from src.modeling import reason_codes as rc
from src.preprocessing_v2 import features as _features  # noqa: F401  (classes the pickled preprocessor needs)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = PROJECT_ROOT / "models" / "final" / "04_scoring_package" / "bundle"
VALID_TERMS = (36, 60)


class ApplicationError(ValueError):
    """The application cannot be scored (missing or impossible values)."""


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raw_field_of(feature, raw_fields):
    """Raw application field a model feature is computed from."""
    if feature.startswith(rc.MISSING_PREFIX):
        return raw_field_of(feature[len(rc.MISSING_PREFIX):], raw_fields)
    for prefix, (concept, _) in rc.ONE_HOT.items():
        if feature.startswith(prefix):
            return concept
    engineered = {"credit_history_months": "earliest_cr_line", "term_months": "term", "emp_length_years": "emp_length"}
    if feature in engineered:
        return engineered[feature]
    if feature in raw_fields:
        return feature
    raise KeyError(f"no raw field found for model feature {feature!r}")


class CreditRiskEngine:
    def __init__(self, bundle_dir=BUNDLE_DIR, strict_versions=True):
        self.bundle_dir = Path(bundle_dir)
        self.manifest = json.loads((self.bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in self.manifest["sha256"].items():
            if file_sha256(self.bundle_dir / name) != expected:
                raise RuntimeError(f"bundle file {name} does not match its checksum in manifest.json")
        fitted_with = self.manifest["environment"]["scikit_learn"]
        if strict_versions and sklearn.__version__ != fitted_with:
            raise RuntimeError(f"scikit-learn {sklearn.__version__} is installed but the preprocessor was fitted with "
                               f"{fitted_with}; install scikit-learn=={fitted_with}")

        files = self.manifest["files"]
        self.preprocessor = joblib.load(self.bundle_dir / files["preprocessor"])
        self.booster = lgb.Booster(model_file=str(self.bundle_dir / files["model"]))
        self.calibrator = json.loads((self.bundle_dir / files["calibrator"]).read_text(encoding="utf-8"))
        self.policy = json.loads((self.bundle_dir / files["policy"]).read_text(encoding="utf-8"))
        self.features = self.manifest["features"]
        self.raw_fields = list(self.preprocessor.named_steps["engineer"].input_columns_)
        self.model_fields = self.manifest["model_input_fields"]
        if self.booster.num_feature() != len(self.features):
            raise RuntimeError("model and manifest disagree on the number of features")
        self._cache_target_encoder()

    def _cache_target_encoder(self):
        """Speed-up only: look target encodings up in dictionaries built once.

        scikit-learn's TargetEncoder.transform rebuilds its category lookup (143,539 job titles) on every
        call, ~0.3 s per application. The fitted values are copied into plain dicts here; unknown
        categories get the training prior, exactly as scikit-learn does. The fresh-session test checks
        the result is identical to the scikit-learn path.
        """
        te = self.preprocessor.named_steps["encode"].named_transformers_["te"].named_steps["encode"]
        lookups = [dict(zip(cats.tolist(), enc.tolist())) for cats, enc in zip(te.categories_, te.encodings_)]
        prior = float(te.target_mean_)

        def fast_transform(X):
            values = np.asarray(X, dtype=object)
            return np.column_stack([[lookup.get(v, prior) for v in values[:, j]] for j, lookup in enumerate(lookups)]).astype(np.float64)

        self._sklearn_te_transform = te.transform
        te.transform = fast_transform

    # ------------------------------------------------------------------ input handling
    def _prepare(self, application):
        """Validate one application; return (clean dict, missing model fields, warnings)."""
        if not isinstance(application, dict):
            raise ApplicationError("an application must be a JSON object / dict")
        app = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in application.items()}
        warnings = [f"field '{k}' is not used by the engine and was ignored" for k in app if k not in self.raw_fields]

        try:
            loan = float(app.get("loan_amnt"))
        except (TypeError, ValueError):
            raise ApplicationError("loan_amnt (loan amount in USD) is required and must be a number") from None
        if not 0 < loan <= 1_000_000:
            raise ApplicationError("loan_amnt must be between 0 and 1,000,000 USD")
        term = pd.Series([app.get("term")]).astype("string").str.extract(r"(\d+)", expand=False).iloc[0]
        if pd.isna(term) or int(term) not in VALID_TERMS:
            raise ApplicationError("term is required and must be 36 or 60 (months)")
        app["term"] = f" {int(term)} months"
        for field in ("annual_inc", "dti", "revol_util", "revol_bal"):
            value = app.get(field)
            if value is not None and float(value) < 0:
                raise ApplicationError(f"{field} cannot be negative")
        fico = app.get("fico_range_low")
        if fico is not None and not 300 <= float(fico) <= 850:
            raise ApplicationError("fico_range_low must be between 300 and 850")

        if not app.get("issue_d"):
            app["issue_d"] = date.today().strftime("%b-%Y")
        if app.get("earliest_cr_line") and pd.isna(pd.to_datetime(str(app["earliest_cr_line"]).strip(), format="%b-%Y", errors="coerce")):
            raise ApplicationError("earliest_cr_line must look like 'Mar-2010'")
        missing = [f for f in self.model_fields if app.get(f) is None or app.get(f) == ""]
        return {k: v for k, v in app.items() if k in self.raw_fields}, missing, warnings

    def transform(self, records):
        """Raw applications (list of dicts or DataFrame) -> the model's 146 input columns."""
        frame = pd.DataFrame(records).reindex(columns=self.raw_fields)
        frame = frame.astype(object).where(frame.notna(), np.nan)
        return self.preprocessor.transform(frame)[self.features]

    # ------------------------------------------------------------------ scoring
    def score_batch(self, records):
        """Many applications at once (no validation, no reasons). Returns a DataFrame."""
        frame = pd.DataFrame(records)
        X = self.transform(frame)
        pd_cal = dl.calibrate(self.booster.predict(X), self.calibrator)
        term = X["term_months"].to_numpy()
        out = dl.apply_policy(pd_cal, frame["loan_amnt"].astype(float).to_numpy(), term, self.policy)
        return pd.DataFrame(out, index=frame.index)

    def score(self, application, n_reasons=4, n_strengths=2):
        """One application -> the full decision as a JSON-ready dict."""
        start = time.perf_counter()
        app, missing, warnings = self._prepare(application)
        X = self.transform([app])
        pd_cal = float(dl.calibrate(self.booster.predict(X), self.calibrator)[0])
        term = int(X["term_months"].iloc[0])
        result = dl.apply_policy([pd_cal], [float(app["loan_amnt"])], [term], self.policy)
        band = str(result["risk_band"][0])
        decision = str(result["decision"][0])
        concepts = rc.concept_shap(self.booster.predict(X, pred_contrib=True), self.features)
        reasons = rc.explain(concepts.iloc[0], X.iloc[0], n_risk=n_reasons, n_protective=n_strengths)
        for item in reasons["risk_factors"] + reasons["protective_factors"]:
            # target-encoded text fields have no missing flag: say so instead of showing a bare rate
            if item["concept"] in ("emp_title", "title") and item["concept"] in missing:
                name = "Job title" if item["concept"] == "emp_title" else "Loan title"
                item["reason"] = f"{name}: not provided (applicants without one default at {float(X[item['concept']].iloc[0]):.1%})"
        t = self.policy["thresholds"]
        rule = {"APPROVE": f"PD <= {t['approve_max_pd']:.1%}", "DECLINE": f"PD >= {t['decline_min_pd']:.1%}",
                "REVIEW": f"{t['approve_max_pd']:.1%} < PD < {t['decline_min_pd']:.1%}"}[decision]
        if missing:
            warnings.append(f"{len(missing)} model input(s) not provided; the training medians / 'missing' "
                            "encodings were used (see missing_fields)")
        return {
            "model": {"name": self.manifest["model_name"], "version": self.manifest["version"]},
            "application_month": app["issue_d"],
            "probability_of_default": round(pd_cal, 4),
            "risk_band": band,
            "risk_band_label": next(b["label"] for b in self.policy["risk_bands"] if b["band"] == band),
            "decision": decision,
            "decision_rule": rule,
            "loan_amount": float(app["loan_amnt"]),
            "term_months": term,
            "expected_loss_usd": round(float(result["expected_loss"][0]), 2),
            "expected_loss_rate": round(float(result["expected_loss_rate"][0]), 4),
            "reasons": [{"reason": r["reason"], "impact": round(r["shap_log_odds"], 4)} for r in reasons["risk_factors"]],
            "strengths": [{"reason": r["reason"], "impact": round(r["shap_log_odds"], 4)} for r in reasons["protective_factors"]],
            "missing_fields": missing,
            "warnings": warnings,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
