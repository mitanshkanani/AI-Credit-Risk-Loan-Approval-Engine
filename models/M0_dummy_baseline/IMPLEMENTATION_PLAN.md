# M0 – Dummy baseline: implementation plan

## Goal
Establish the **floor** every real model must beat. M0 ignores all 190 features and gives every
applicant the same probability of default (PD): the training default rate. If a later model barely
beats M0, something is wrong.

## What M0 is
`sklearn.dummy.DummyClassifier(strategy="prior")`
- `fit` only learns the class frequencies in `y_train` (17.24% defaults).
- `predict_proba` returns `[0.8276, 0.1724]` for **every** applicant.
- It has no hyper-parameters to tune and does not look at X at all.

## Data used
| Split | Used for | Notes |
|---|---|---|
| `final_preprocessed_data_v2/y_train.parquet` | fit + report | 388,124 rows, 17.24% default |
| `y_validation.parquet` | report | 162,745 rows, 20.35% default |
| `X_*.parquet` → column `term_months` only | 36 vs 60-month segment report | features are not used by the model |
| **test set** | **not touched** | sealed until the final model is chosen (ground rule from the model strategy) |
| stress 2016-2018 | not used | only for the final model |

## Steps (one notebook section each)
1. **Setup**: locate the repo root, import the shared evaluation module, print versions.
2. **Load data**: y for train / validation plus `term_months`; assert row counts and default rates match `artifacts_v2/metadata_v2.json`.
3. **Fit M0** on the training labels; show the learned class prior.
4. **Predict** PD for train and validation.
5. **Score** with the standard report card used by every model:
   ROC-AUC, PR-AUC, KS, Brier score, log loss, mean PD vs actual default rate (calibration gap).
6. **Accuracy trap**: show that M0 gets about 80% accuracy while catching 0% of defaults, which is why accuracy is not used.
7. **Calibration**: predicted vs actual default rate per split (shows the 17% → 20% drift).
8. **Approval-rate vs bad-rate curve**: with no ranking ability, approving any share of applicants gives the overall bad rate.
9. **Segments**: the same metrics for 36- and 60-month loans.
10. **Version A vs B**: identical for M0 because it uses no features; recorded once for both.
11. **Save**: model, metrics JSON, segment and curve tables, plots, and a row per split in `models/leaderboard.csv`.
12. **Conclusion**: the numbers M1 must beat.

## Expected results (known before running; verified in the notebook)
- ROC-AUC = **0.500**, KS = **0.000** (no ranking at all)
- PR-AUC ≈ default rate of the split (≈ 0.172 train, ≈ 0.204 validation)
- Brier (validation) = r(1−p)² + (1−r)p² with p = 0.1724, r = 0.2035 ≈ **0.163**
- Validation calibration gap ≈ 0.172 − 0.204 = **−3.1 points** (under-predicts because of drift)

## Files produced
```
models/
├── leaderboard.csv                          # one row per model × version × split (shared by all models)
└── M0_dummy_baseline/
    ├── IMPLEMENTATION_PLAN.md               # this file
    ├── M0_dummy_baseline.ipynb              # executed notebook (cell outputs saved)
    ├── M0_dummy_baseline.pdf                # 1-2 page summary
    ├── build_m0_pdf.py                      # generates the PDF from the results
    └── results/
        ├── m0_metrics.json
        ├── m0_segment_metrics.csv
        ├── m0_approval_curve.csv
        ├── m0_calibration.png
        ├── m0_approval_curve.png
        └── m0_dummy.joblib                  # not tracked by git (*.joblib)
src/modeling/evaluation.py                   # shared report card reused by M1-M5
```

## Execution
- Notebook executed in Jupyter (Anaconda Python 3.13.9) with
  `jupyter nbconvert --to notebook --execute --inplace`, so all cell outputs are stored.
- PDF built with ReportLab from the saved results.

## Known caveat for later models
Anaconda has scikit-learn 1.7.2, but `preprocessor_v2.joblib` was fitted with 1.9.0.
M0 doesn't load the preprocessor, so this doesn't matter yet. Before deployment, the
preprocessor and the models must run in one environment with the same scikit-learn version.
