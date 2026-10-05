# M1 – Logistic Regression: implementation plan

## Goal
Build the first real model: an explainable, industry-standard baseline that
1. sets the accuracy bar the tree models (M2-M4) must beat,
2. checks for leakage (a linear model scoring far above about 0.75-0.80 ROC-AUC would be suspicious),
3. shows which features push default risk up or down, and whether that matches credit intuition.

## What Logistic Regression does
It learns one weight per feature. For an applicant it adds up `weight × feature value`, then squashes
the total through the sigmoid function into a probability between 0 and 1:

`PD = 1 / (1 + exp(-(b0 + w1·x1 + ... + wn·xn)))`

A positive weight raises the PD, and a negative one lowers it. Because features are standardised, each
weight says how much one standard deviation of that feature changes the **log-odds** of default.

## Data
| Split | Use | Rows / default rate |
|---|---|---|
| train | fit | 388,124 / 17.24% |
| validation | choose hyper-parameters, report | 162,745 / 20.35% |
| test | **sealed** | not loaded |

- **Version A:** all 190 features minus `fico_range_high` = **189** inputs.
- **Version B:** additionally drops the 42 LendingClub risk columns (`grade_*`, `sub_grade_*`, `int_rate`,
  `installment`) = **147** inputs. **B is the deployable model.**

## Model-specific preprocessing (inside the model pipeline, fitted on train only)
Profiling the training data (done before writing this plan) found:
- 126 binary columns (one-hot and missing flags) and 64 continuous columns, none negative.
- 28 continuous columns are heavily right-skewed (skew > 2 on train), e.g. `tot_coll_amt`, `annual_inc`, `revol_bal`.
- `fico_range_low` and `fico_range_high` correlate at **1.000**, so `fico_range_high` is dropped.

Pipeline: `ColumnTransformer` [ `log1p` → `StandardScaler` on the skewed columns | `StandardScaler` on all
other columns ] → `LogisticRegression`. Target-encoded columns (bounded default rates) are only scaled.

## Hyper-parameter search (validation ROC-AUC)
| Setting | Values tried |
|---|---|
| preprocessing | scale only · log1p + scale |
| `C` (inverse regularisation strength) | 0.0001, 0.001, 0.01, 0.1, 1 |
| `class_weight` | None · "balanced" |
| fixed | `penalty="l2"`, `solver="lbfgs"`, `max_iter=3000` |

20 fits per version (about 3-5 s each). **Selection rule:** highest validation ROC-AUC. Among
configurations within 0.0005 of the best, prefer `class_weight=None` (honest probabilities),
then the smallest `C` (simplest model).

## Evaluation (shared report card, `src/modeling/evaluation.py`)
ROC-AUC, PR-AUC, KS, Brier, log loss, calibration gap; decile calibration plot; approval-rate vs
bad-rate curve compared with M0; 36- vs 60-month segments; A minus B gap.

## Sanity checks in the notebook
- Solver converged (`n_iter_ < max_iter`).
- Validation ROC-AUC above M0 (0.500) and below the 0.80 leakage flag.
- Coefficient signs match credit intuition: FICO ↓ risk, DTI ↑, 60-month term ↑, income ↓,
  interest rate ↑ (A only).

## Files produced
```
models/M1_logistic_regression/
├── IMPLEMENTATION_PLAN.md
├── M1_logistic_regression.ipynb      # executed in Jupyter, outputs saved
├── M1_logistic_regression.pdf        # 2-page summary
├── build_m1_pdf.py
└── results/
    ├── m1_tuning_results.csv          # all 40 configurations
    ├── m1_metrics.json                # chosen A and B: parameters + report cards
    ├── m1_coefficients.csv            # standardised weights + odds ratios (A and B)
    ├── m1_segment_metrics.csv
    ├── m1_approval_curve.csv
    ├── m1_calibration.png, m1_approval_curve.png, m1_coefficients.png
    └── m1_A.joblib, m1_B.joblib       # not tracked by git
```
Leaderboard rows: M1 / A and M1 / B for train and validation.
