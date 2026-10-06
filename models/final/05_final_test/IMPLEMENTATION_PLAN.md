# Stage 5 – Final test (opened once): implementation plan

Written **before** the test split is opened. The engine is frozen: bundle v1.0.0, commit 9d55a32
(manifest checksums). **Nothing is changed after this stage**, whatever the results. Results are reported
as they come out, good or bad.

## Data
- **Test:** 2015 H2, 212,801 loans, never used for any choice so far.
- **Stress (optional):** 2016–2018, 518,744 loans. Labels are biased (many 2017–2018 loans had not
  finished when the data ends, and those that finished early are mostly defaults or early payoffs),
  so the stress set is used for ranking (ROC-AUC) only, not for calibration or default rates.

## How it is scored
1. All 212,801 test loans: `X_test` (already produced by the same fitted preprocessor) → bundle model
   → bundle calibrator → bundle policy (`src/modeling/decision.py`).
2. End-to-end check: 5,000 random test loans scored by `CreditRiskEngine` from their **original raw CSV
   rows** must give the same PDs as step 1 (proves the numbers belong to the deployable engine).

## Pre-registered criteria (reported, not used to change anything)
| Criterion | Target |
|---|---|
| ROC-AUC drop vs validation (0.7351) | ≤ 0.02 |
| Calibration gap (mean PD − default rate) | within ±2 points |
| 36- and 60-month calibration gaps | within ±2 points (watch 60-month loans, ~1 point low on validation) |
| Risk bands: default rate rises band to band | required |
| Approved book default rate | ≤ 13% (validation target 12% + 1 point) |
| Expected loss ÷ realised loss | within 0.85–1.15 |
| End-to-end raw CSV check | max PD difference < 1e-9 |

## Also reported
All metrics (ROC-AUC, PR-AUC, KS, Brier, log loss, ECE), decision mix, band table, business impact,
state calibration (flagged states from Stage 2: LA, TN, OR), and a benchmark: how well LendingClub's
own interest rate (which is set from its internal grade) ranks the same loans. Our model never sees the
interest rate (version B).

Runs locally (scoring only, no training).
