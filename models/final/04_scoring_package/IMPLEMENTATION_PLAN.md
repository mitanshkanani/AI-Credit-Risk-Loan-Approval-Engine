# Stage 4 – Scoring package: implementation plan

**Goal:** one loadable engine. A raw loan application goes in; the final JSON comes out (PD, risk band,
decision, expected loss, reasons). This is exactly what Stage 5 (final test) and Stage 6 (live API)
will call.

```
raw application (dict / JSON)
  -> preprocessor_v2.joblib    feature engineering, imputation, encoding (fitted on TRAIN)
  -> final_model.txt           LightGBM M3-B-mono-nozip, 146 features (Stage 2)
  -> calibrator.json           Platt scaling (Stage 2)
  -> decision_policy.json      bands, approve/review/decline, expected loss (Stage 3)
  -> reason codes              SHAP -> plain-language reasons (Stage 2)
  -> JSON
```

## Design
- **Bundle** `bundle/`: the 4 files above plus `manifest.json` (SHA-256 checksums, library versions,
  the 146 features in model order, the 65 raw input fields that matter, the ignored raw fields).
  Built by `build_bundle.py`. The bundle files are small (~10 MB) and are committed so the live demo
  can be deployed straight from GitHub.
- **Engine** `src/engine/scorer.py` (`CreditRiskEngine`):
  - refuses to start if a bundle file does not match its checksum, or if scikit-learn is not the
    version that fitted the preprocessor (1.9.0);
  - validates input (loan amount, term 36/60, FICO 300–850, no negative income/DTI/balances,
    credit-line date format) and raises `ApplicationError` with a readable message;
  - fields not supplied are allowed: they get the TRAIN medians / "missing" encodings, and the
    output lists them in `missing_fields`;
  - `score(dict)` → full JSON for one applicant; `score_batch(rows)` → vectorised PD, band,
    decision, expected loss for many.
- **One environment:** Python 3.14 with scikit-learn 1.9.0 (same as the preprocessor fit),
  LightGBM 4.7, pinned in `requirements-engine.txt`.

## Fresh-session test (`tests/engine_fresh_session_test.py`, run as its own process)
Required:
1. Bundle loads and all checksums match; a tampered file is refused.
2. Original raw CSV rows for 2,000 **validation** loans reproduce Stage 3's PD, band, decision and
   expected loss exactly (test split stays sealed).
3. Single-application scoring agrees with batch scoring.
4. A sparse hand-written application gives complete JSON without NaN.
5. Impossible inputs are rejected with a message.
6. Same input → same output; higher FICO never raises PD; higher DTI never lowers it.

Warning only: one application scored in under 200 ms.

## Outputs (`models/final/04_scoring_package/`)
`bundle/`, `build_bundle.py`, `requirements-engine.txt`, executed notebook `F4_scoring_package.ipynb`,
2-page PDF, `results/` (test report, example outputs, latency breakdown).
