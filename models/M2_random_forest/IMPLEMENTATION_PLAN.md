# M2 – Random Forest: implementation plan

## Goal
Answer one question from the model strategy: **do non-linear effects and feature interactions add
real signal over Logistic Regression (M1)?**
- Clear gain over M1 → non-linearity matters, and boosting (M3/M4) is worth the effort.
- No gain → the signal is mostly linear; M1 stays a strong, explainable candidate.

## What a Random Forest does
It grows hundreds of decision trees. Each tree:
1. trains on a random **bootstrap sample** of the loans,
2. at every split, looks at only a random subset of the features (`max_features`),
3. keeps splitting (e.g. "FICO < 690?" then "DTI > 25?") until a leaf would hold fewer than `min_samples_leaf` loans.

A leaf's PD is the default rate of the training loans that landed in it. The forest's PD is the
**average over all trees**. Averaging many different, noisy trees cancels out their individual errors.
Trees need **no scaling or log transforms**, and they find interactions such as "high DTI is only
dangerous with low income" by themselves.

## Data and versions (same feature sets as M1 for a fair comparison)
- Version A: 189 features (all 190 minus `fico_range_high`).
- Version B: 147 features (also without grade / sub_grade / int_rate / installment). **Deployable.**
- Fit on train (388,124), choose settings on validation (162,745), **test sealed**.
- Features are cast to float32 (what scikit-learn's trees use internally) to save memory.

## Machine constraints (measured before planning)
12 logical cores, **about 3.3 GB free RAM**. A 50-tree forest (version B) takes 14 s with
`max_features="sqrt"` and 48 s with `max_features=0.3`, at about 7,300 nodes per tree.
So:
- `max_samples=0.5`: each tree sees a random half of the training loans. This halves time and memory
  and adds diversity between trees; it is a standard Random Forest setting.
- Tuning uses 200 trees; the chosen setting is refit with 500 trees.
- Only one forest is kept in memory at a time: each is scored, saved, then deleted.

## Hyper-parameter search (validation ROC-AUC)
| Setting | Values tried | Meaning |
|---|---|---|
| `min_samples_leaf` | 20, 50, 100 | minimum loans per leaf; larger = smoother, less overfitting |
| `max_features` | "sqrt" (about 12-14 features), 0.3 (30% of features) | features tried per split; fewer = more diverse trees |
| `class_weight` | None; "balanced_subsample" checked on the best setting | up-weights defaults per bootstrap sample |
| fixed | `n_estimators=200` (tuning) → 500 (final), `max_samples=0.5`, `bootstrap=True`, `criterion="gini"`, `random_state=42`, `n_jobs=-1` | |

7 fits per version (6 grid + 1 class-weight check). **Selection rule:** highest validation ROC-AUC.
Within 0.0005 of the best, prefer `class_weight=None`, then the larger `min_samples_leaf`
(simpler), then `"sqrt"` (faster).

## Evaluation
- Shared report card (ROC-AUC, PR-AUC, KS, Brier, log loss, calibration gap) on train and validation.
- **Direct comparison with M1 (same version):** the main output of M2.
- **Number-of-trees curve:** validation AUC using the first 25, 50, ..., 500 trees (shows when more trees stop helping).
- **Feature importance:** impurity-based (MDI) top 15, plus permutation importance (drop in validation
  AUC when a feature is shuffled) for the top 20 on a 20,000-row validation sample. MDI is biased toward
  continuous features; permutation importance is the more honest check.
- Calibration deciles, approval-rate vs bad-rate curve (M0, M1-B, M2-B), 36 vs 60-month segments, A minus B gap.
- Sanity: beats M0, below the 0.80 leakage flag, train vs validation gap (forests can overfit train).

## Where it runs: Kaggle (private)
The first local run was stopped by the operating system's low-memory protection (about 3.3 GB free on
this 16 GB laptop). M2 therefore runs on a **private Kaggle CPU notebook** (about 30 GB RAM, 4 cores),
driven entirely from the Kaggle CLI:
1. A private dataset `mitanshkanani/credit-risk-m2-inputs` holds only **train + validation**
   X/y Parquet files (test and stress files are not uploaded), `metadata_v2.json`, `evaluation.py`,
   `leaderboard.csv`, M1-B's validation predictions and this notebook.
2. A private runner kernel (`kaggle/runner.py`) rebuilds the repo layout under `/kaggle/working/repo`
   and executes `M2_random_forest.ipynb` with `jupyter nbconvert --execute`, so the executed notebook,
   with every cell output, is an output file.
3. `kaggle kernels output` downloads the executed notebook and `results/` back into this folder,
   and the PDF is built locally.

The notebook is identical locally and on Kaggle. Its only Kaggle-aware line points the data loader at
the input dataset. On Kaggle the 500-tree forests (hundreds of MB each) are not saved: their
validation predictions are kept instead (`m2_validation_pd.parquet`), and a refit is reproducible
(`random_state=42`).

## Files produced
```
models/M2_random_forest/
├── IMPLEMENTATION_PLAN.md
├── M2_random_forest.ipynb          # executed in Jupyter, outputs saved
├── M2_random_forest.pdf            # 2-page summary
├── build_m2_pdf.py
└── results/
    ├── m2_tuning_results.csv, m2_metrics.json, m2_tree_curve.csv
    ├── m2_feature_importance.csv, m2_segment_metrics.csv, m2_approval_curve.csv
    ├── m2_calibration.png, m2_approval_curve.png, m2_importance.png, m2_tree_curve.png
    ├── m2_validation_pd.parquet    # validation PDs of M2-A and M2-B
    └── m2_A.joblib, m2_B.joblib    # only when run locally; not tracked by git (large)
├── kaggle/                          # dataset + runner kernel metadata used for the Kaggle run
```
Leaderboard rows: M2 / A and M2 / B for train and validation.
