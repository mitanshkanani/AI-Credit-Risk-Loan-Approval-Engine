# Stage 3 – Decision layer: implementation plan

**Starting point (Stage 2):** final model M3-B-mono-nozip (146 features) + Platt calibrator
(a = 1.0386, b = 0.1685). The model outputs a calibrated probability of default (PD). This stage turns
that PD into what a lender acts on: a **risk band**, a **decision** (approve / review / decline) and
an **expected loss** in dollars.

Validation split only (2015 H1, 162,745 loans). **Test stays sealed.** Runs locally (light work:
one LightGBM prediction pass, no training).

## Inputs
- `models/final/02_reason_codes_fairness/results/final_model.txt`, `calibrator.json`
- `results/loss_table.parquet`, extracted from the raw CSV by `extract_loss_table.py` (train and
  validation loans only): funded amount, principal repaid, recoveries, collection fees. These are
  post-loan outcomes, never model features; they are used only to measure losses.

## Rules (fixed before looking at validation results)

1. **Loss model, from defaulted TRAINING loans only (2012-08..2014-12)**
   - EAD (exposure at default) = funded amount − principal repaid before default.
     EAD share = total EAD ÷ total funded amount, per term (36 / 60 months).
   - LGD (loss given default) = 1 − (recoveries − collection fees) ÷ EAD, pooled per term.
   - Expected loss of an applicant: **EL = PD × loan amount × EAD share × LGD**.
2. **Risk bands:** 7 fixed bands on calibrated PD: R1 < 5%, R2 5–10%, R3 10–15%, R4 15–20%,
   R5 20–30%, R6 30–40%, R7 ≥ 40%. Check: the observed default rate must rise from band to band.
3. **Business targets for the decision thresholds** (searched on a 0.5-point PD grid, using
   out-of-fold calibrated PDs so the thresholds are not tuned on in-sample probabilities):
   - **APPROVE** if PD ≤ the highest threshold at which the approved book's observed default rate is
     **≤ 12%** (about 40% below the 20.35% of all validation loans).
   - **DECLINE** if PD ≥ the lowest threshold above which applicants' observed default rate is
     **≥ 40%** (two in five default).
   - **REVIEW** (manual underwriting) in between.
4. **Checks**
   - Bands: observed default rate strictly increasing (required).
   - Expected-loss back-test: predicted EL on validation within ±10% of the realised loss (required).
     Note: recoveries on 2015 loans were still being collected when the data ends (2018 Q4), so
     realised loss may be slightly overstated.
   - Review queue ≤ 35% of applicants (warning only; it is a staffing question, not a model one).

## Outputs (`models/final/03_decision_layer/`)
- `src/modeling/decision.py`: applies the policy (band, decision, EL); reused by Stages 4–6.
- `results/decision_policy.json`: every number of the policy in one plain file.
- Executed notebook `F3_decision_layer.ipynb`, 2-page PDF `F3_decision_layer.pdf`, tables and plots
  in `results/`.
