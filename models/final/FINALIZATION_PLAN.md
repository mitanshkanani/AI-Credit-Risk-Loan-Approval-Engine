# Finalizing the credit-risk engine (after the M0-M4 model ladder)

**Winner of the ladder:** M3 LightGBM, version B (no LendingClub grade features; deployable).
Validation ROC-AUC 0.7362, Brier 0.1426. XGBoost (M4) tied, which confirms the result; LightGBM is
about 2.3× faster. The monotonic variant M3-B-mono costs only 0.0008 ROC-AUC.

The finalization runs in stages. The user reviews each stage before the next one starts. **The test
split stays sealed until Stage 5**, and every decision is frozen before it is opened.

| Stage | What | Key decision rule (fixed before looking at results) |
|---|---|---|
| **1. Model choice + calibration** | Choose M3-B vs M3-B-mono; verify the monotonic constraints really hold; calibrate the probabilities | Choose B-mono if it costs < 0.005 ROC-AUC (tie margin) **and** has 0 monotonicity violations. Calibration method = lowest out-of-fold Brier on validation (tie → simpler Platt) |
| 2. Reason codes + fairness | Turn SHAP values into plain-language reasons (top factors per applicant); check results across states and zip regions; measure what `zip_code` adds | Drop `zip_code` if it adds < 0.002 ROC-AUC (fair-lending risk is not worth a tiny gain) |
| 3. Decision layer | Risk bands, approve / review / decline thresholds, expected loss (PD × LGD × exposure), with LGD estimated from recoveries on **defaulted training loans** | Thresholds chosen on validation from an explicit business target |
| 4. Scoring package | One loadable engine: preprocessor → model → calibrator → decision layer, in one consistent environment (scikit-learn version aligned) | Fresh-session test: raw application in, final JSON out |
| 5. Final test (once) | Open the sealed 2015 H2 test set, score the frozen engine, report every metric; optional 2016-2018 stress check | Nothing may change after this |

Light work (loading saved models, calibration, SHAP, decision rules) runs locally in Jupyter; any
heavy retraining runs on Kaggle, as with M2-M4.

---

# Stage 1 – Model choice and calibration: implementation plan

## Inputs
- `models/M3_lightgbm/results/m3_B.txt`, `m3_B_mono.txt` (saved LightGBM models)
- Validation split only (162,745 loans, 2015 H1). Test stays sealed.

## Steps
1. **Integrity check:** reload both saved models locally and confirm their validation predictions match
   the PDs saved on Kaggle (`m3_validation_pd.parquet`).
2. **Monotonicity check:** for 1,000 random validation applicants, sweep each of the 6 constrained
   features (FICO, DTI, income, term, inquiries, revolving utilisation) across its observed range
   while holding everything else fixed, and count applicants whose PD ever moves the "wrong" way.
   Done for B (unconstrained) and B-mono.
3. **Model choice** by the rule above.
4. **Calibration** of the chosen model on validation, comparing:
   - none (raw LightGBM PD),
   - **Platt scaling**: a logistic regression on the logit of the PD (2 numbers),
   - **isotonic regression**: a step-shaped, monotone mapping.

   Each is evaluated **out-of-fold** (5-fold cross-fitting inside validation), so the calibration
   score is not measured on the same loans it was fitted on. Metrics: Brier, log loss, calibration gap,
   expected calibration error (ECE), and the gap for 36- and 60-month loans.
5. **Fit the chosen calibrator on all of validation** and save it as plain JSON (Platt: 2 coefficients;
   isotonic: breakpoints). It is version-independent and has no scikit-learn pickle.

## Outputs (`models/final/01_selection_calibration/`)
`IMPLEMENTATION_PLAN.md` (this section), executed notebook, 2-page PDF, `results/`
(`final_model_choice.json`, `calibrator.json`, monotonicity and calibration tables, plots).
