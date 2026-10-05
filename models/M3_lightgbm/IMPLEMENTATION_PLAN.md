# M3 – LightGBM (gradient boosting): implementation plan

## Goal
Find the best accuracy this data supports. M3 is the **main candidate** in the model strategy.
M2 showed that averaging independent trees adds almost nothing over Logistic Regression;
boosting is the real test of whether non-linear effects matter.

## What gradient boosting does
Trees are built **one after another**. Each new small tree is fitted to the mistakes (gradients
of the log loss) that all previous trees together still make, and its output is added with a small
weight (the *learning rate*). After a few hundred rounds the sum of trees gives the log-odds of
default, and the sigmoid turns it into a PD. Unlike a Random Forest (independent deep trees,
averaged), boosting uses many **shallow, cooperating** trees, which usually wins on tabular data.
LightGBM is a fast implementation: it bins features into histograms and grows trees leaf-wise.

## Early evidence (measured before planning)
One untuned LightGBM on version B (learning rate 0.05, 31 leaves) reached **validation ROC-AUC
0.7347** vs 0.7252 (M1-B) and 0.7270 (M2-B), in about 20 s on 4 threads (539 boosting rounds).

## Data and versions (same as M1/M2)
- A: 189 features. B: 147 (no LendingClub grade / sub_grade / int_rate / installment). **B is deployable.**
- Fit on train (388,124), early-stop and choose on validation (162,745), **test sealed**.
- Note: validation is used both to stop boosting and to choose settings, so its score is slightly
  optimistic. The sealed test set gives the honest final number for the chosen model.

## Training setup
LightGBM core API (`lgb.train`), objective `binary`, early stopping after 100 rounds without
validation-AUC improvement (max 5,000 rounds), `bagging_freq=1`, `seed=42`, all CPU threads.

## Hyper-parameter search: random search, 40 settings per version
| Parameter | Range | Meaning |
|---|---|---|
| `num_leaves` | 15-255 (log scale) | tree complexity |
| `min_data_in_leaf` | 20-2,000 (log scale) | minimum loans per leaf (regularisation) |
| `feature_fraction` | 0.4-1.0 | share of features per tree |
| `bagging_fraction` | 0.5-1.0 | share of loans per tree |
| `lambda_l1`, `lambda_l2` | 1e-4-10 (log scale) | weight penalties |
| `min_gain_to_split` | 0-0.5 | minimum improvement to split |
| fixed | `learning_rate=0.05` | |

The same 40 settings are tried for A and B (seed 42); trial 0 is the untuned baseline above.
**Selection rule:** highest validation ROC-AUC; within 0.0005, prefer fewer leaves (simpler).

## Follow-up experiments on the chosen setting
1. **Lower learning rate (0.02)**: kept only if validation ROC-AUC improves by at least 0.0005.
2. **Class imbalance**: `is_unbalance=True`; expected to keep AUC and hurt probabilities (as in M1/M2).
3. **Monotonic constraints (B)**: force FICO ↓, DTI ↑, income ↓, 60-month term ↑, recent inquiries ↑,
   revolving utilisation ↑. This makes the model safer to defend. The AUC cost is measured and
   recorded as leaderboard version `B-mono`.

## Evaluation
- Shared report card (ROC-AUC, PR-AUC, KS, Brier, log loss, calibration gap) for train and validation.
- **Comparison with M1 and M2** (same version), with the +0.005 "clear gain" rule.
- Learning curve (validation AUC per boosting round), decile calibration, approval-rate vs bad-rate
  curve (M0, M1-B, M2-B, M3-B), 36 vs 60-month segments, A minus B gap.
- **SHAP explanations** (LightGBM's built-in `pred_contrib`, 20,000 validation loans): global
  mean |SHAP| ranking, plus one example applicant's top reasons, a preview of the reason codes the
  decision layer will return.

## Where it runs: Kaggle (private), same pattern as M2
- Private dataset `mitanshkanani/credit-risk-model-inputs`: train + validation X/y (no test or
  stress), `metadata_v2.json`, M1-B and M2 validation predictions. It can be reused by M4.
- Private script kernel `mitanshkanani/credit-risk-m3-lightgbm`: `kaggle/runner.py` embeds the
  notebook, `evaluation.py` and the current leaderboard, rebuilds the repo layout and executes the
  notebook with `nbconvert`.
- Results come back with `kaggle kernels output`; the PDF is built locally.
- A 2-trial smoke test runs locally first.

## Files produced
```
models/M3_lightgbm/
├── IMPLEMENTATION_PLAN.md, M3_lightgbm.ipynb (executed on Kaggle), M3_lightgbm.pdf, build_m3_pdf.py
├── kaggle/  build_runner.py, runner.py, kernel-metadata.json, dataset-metadata.json
└── results/
    ├── m3_tuning_results.csv, m3_metrics.json, m3_experiments.csv
    ├── m3_shap_importance.csv, m3_example_explanation.csv, m3_segment_metrics.csv, m3_approval_curve.csv
    ├── m3_learning_curve.png, m3_calibration.png, m3_approval_curve.png, m3_shap.png
    ├── m3_A.txt, m3_B.txt, m3_B_mono.txt   # LightGBM model files (text, small)
    └── m3_validation_pd.parquet            # gitignored
```
Leaderboard rows: M3 / A, M3 / B, M3 / B-mono for train and validation.
