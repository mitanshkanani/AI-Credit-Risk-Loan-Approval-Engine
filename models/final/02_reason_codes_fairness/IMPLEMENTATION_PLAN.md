# Final model · Stage 2 – Fairness checks and reason codes: implementation plan

## Starting point (from Stage 1)
Final model **M3-B-mono** (LightGBM, 147 features, monotonic constraints) + Platt calibration
(`calibrator.json`). Only the **validation** split is used; the test split stays sealed.

## Part A – Fairness (done first, because it can change the final model)
The data has no protected attributes (no race, sex or age), so we check the main **proxies**:
geography (`zip_code`, `addr_state`) and how the model treats different areas.

1. **Zip-code ablation (decision rule fixed now):** retrain M3-B-mono with exactly the same settings
   (learning rate 0.02, the chosen M3 parameters, monotonic constraints, early stopping on validation)
   - (a) without `zip_code`,
   - (b) without `zip_code` and `addr_state` (all geography; for information).

   **Rule:** if `zip_code` adds **less than 0.002 ROC-AUC**, drop it. A tiny gain is not worth the
   fair-lending risk of using location as a proxy for protected groups. If it is dropped, the new model
   is recalibrated (same out-of-fold Platt procedure) and its monotonicity is re-verified.
   These are single LightGBM fits (about 1-3 minutes each), so they run locally.
2. **Who is affected by zip codes:** split applicants into 5 groups by their area's historical default
   rate. Under a reference policy (approve the safest 70%; only a diagnostic until Stage 3 sets real
   thresholds), compare approval rates per group with and without `zip_code`, and count how many
   decisions flip.
3. **State-level check (final model):** for every state with at least 1,000 validation loans, compare
   loans, actual default rate, mean calibrated PD (out-of-fold), calibration gap, ROC-AUC, approval rate
   and bad rate among the approved. **Fairness criterion:** the same PD should mean the same risk in
   every state. Flag states with a calibration gap beyond ±2 percentage points.

## Part B – Reason codes (on the final model from Part A)
1. SHAP values for all 162,745 validation loans (LightGBM `pred_contrib`), checked to add up exactly
   to the model's log-odds.
2. Columns grouped into **concepts** (`src/modeling/reason_codes.py`): one-hot dummies → one concept
   (e.g. all `addr_state_*` → "State"), missing flags → their feature, so reasons talk about
   "State: CA", not `addr_state_CA`.
3. Plain-language reasons per applicant: up to **4 factors that raised the risk** (the usual number
   of principal reasons in US adverse-action notices) and 2 that lowered it, each with the applicant's
   value (e.g. "Loan term: 60 months", "Credit score (FICO): 660", "Annual income: $54,000").
4. **How often each reason appears** among the riskiest 30% of applicants (those most likely to be
   declined): this shows which reasons customers would actually see.
5. Three worked examples (low, medium and high PD) showing the full output.

## Outputs (`models/final/02_reason_codes_fairness/`)
Executed notebook, 2-page PDF, `results/`: ablation table, zip-group and state tables, reason frequency
table, example explanations, plots, an updated `final_model_choice.json` (plus a new model file and
calibrator only if `zip_code` is dropped).
