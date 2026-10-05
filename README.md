<div align="center">

# 🏦 AI Credit Risk & Loan Approval Engine

**An end-to-end machine-learning engine that estimates the probability that a loan applicant will default,<br/>and turns it into an explainable approve / review / decline decision.**

![Python](https://img.shields.io/badge/Python-3.13%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Data](https://img.shields.io/badge/data-LendingClub%202007--2018-2e9b5b)
![Audits](https://img.shields.io/badge/preprocessing%20audits-72%2F72%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-modeling%20next-orange)

[Pipeline diagram](docs/preprocessing_architecture.svg) ·
[Model strategy (PDF)](docs/model_strategy.pdf) ·
[Preprocessing report](artifacts_v2/preprocessing_v2_report.md)

</div>

---

## 📌 Overview

Lenders have to decide, at the moment an application arrives, whether a borrower is likely to pay the loan back.
This project builds that decision engine on **2.26 million real LendingClub loans (2007-2018)**:

1. **Credit-risk model**: predicts the **probability of default (PD)** for a new application.
2. **Decision layer**: turns the PD into a risk band, an approve / review / decline decision, an expected loss, and the reasons behind it.
3. **Web deployment** (planned): a form on a website goes in, and an explainable decision comes out.

The work is done the way a real credit-risk team would do it. Every column is checked for **data leakage**, the model is evaluated on **later loans than it was trained on**, and every preprocessing step is **verified by automated audits**.

## ✨ Highlights

- 🔍 **Leakage-audited features.** All 151 raw columns were classified as *known at application time* vs *only known after the loan starts*. 42 future-information columns (payments, recoveries, settlements, last FICO, ...) were removed.
- 🕰️ **Out-of-time evaluation.** Train on loans issued Aug-2012 to Dec-2014, validate on 2015 H1, test on 2015 H2. Cohorts that were still largely unfinished (2016-2018) are kept apart as a stress test.
- 🧩 **One reusable pipeline.** Feature engineering, imputation, encoding and column selection live in a single fitted scikit-learn `Pipeline` (`preprocessor_v2.joblib`), fitted on training data only.
- 🎯 **Leak-free target encoding.** High-cardinality fields (`zip_code`, `emp_title`, `title`) are target-encoded with 5-fold cross-fitting; unseen categories fall back to the training default rate.
- ✅ **72 automated audits + a fresh-session inference test.** Covers target alignment, temporal ordering, leakage, missing / infinite / non-numeric values, feature alignment and artifact persistence. Raw applications reproduce the exported features exactly.

## 🗺️ Project status

- [x] Data inspection and EDA reports
- [x] Target definition and eligible population
- [x] Column-by-column leakage audit
- [x] Preprocessing V2: temporal split, fitted pipeline, 72/72 audits passing
- [x] Architecture diagram and model strategy document
- [ ] Baseline models: dummy, Logistic Regression
- [ ] Tree ensembles: Random Forest, LightGBM, XGBoost (each with and without LendingClub grade features)
- [ ] Calibration, SHAP explanations, fairness checks, decision layer
- [ ] Scoring API and website
- [ ] Part 2: rejected-applications model

## 📊 Dataset

[LendingClub loan data 2007-2018Q4](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (Kaggle).
The raw files are **not** included in this repository. Put them in `data/`:

```
data/accepted_2007_to_2018Q4.csv   # 2,260,701 loans × 151 columns
data/rejected_2007_to_2018Q4.csv   # rejected applications (Part 2)
```

**Target definition** (only loans with a final outcome are used):

| `target` | Loan status | Loans |
|:---:|---|---:|
| **0** (repaid) | Fully Paid · Does not meet the credit policy: Fully Paid | 1,078,739 |
| **1** (default) | Charged Off · Default · Does not meet the credit policy: Charged Off | 269,360 |
| *excluded* | Current · Late · In Grace Period (no final outcome yet) | 912,602 |

## 🔧 Preprocessing pipeline

```mermaid
flowchart TD
    A[Raw LendingClub data<br/>2.26M loans × 151 columns] --> B[Define target &<br/>eligible population<br/>1.35M resolved loans]
    B --> C[Column audit &<br/>application-time vs future-time review]
    C --> D[Remove target, IDs &<br/>42 leakage columns<br/>→ 103 candidates]
    D --> E[Quality checks &<br/>missing-value strategy]
    E --> F[Restore id / issue date / target<br/>0 mismatches on 12 check columns]
    F --> G[Temporal split<br/>train · validation · test · stress]
    G --> H[Feature engineering<br/>credit history months, term, employment length]
    H --> I[Fitted pipeline on TRAIN only<br/>column filter · median imputation + missing flags<br/>one-hot · cross-fitted target encoding · de-duplication]
    I --> J[190 model-ready features<br/>X / y Parquet per split]
    J --> K[72 audits + fresh-session inference test ✅]
```

👉 **Full step-by-step explanation:** [`docs/preprocessing_architecture.svg`](docs/preprocessing_architecture.svg)

### Temporal split

| Split | Issue months | Loans | Defaults | Default rate |
|---|---|---:|---:|---:|
| Train | 2012-08 → 2014-12 | 388,124 | 66,928 | 17.24% |
| Validation | 2015-01 → 2015-06 | 162,745 | 33,120 | 20.35% |
| Test | 2015-07 → 2015-12 | 212,801 | 42,684 | 20.06% |
| Stress test *(optional)* | 2016-01 → 2018-12 | 518,744 | 116,295 | 22.42% |

> **Why not test on 2016+?** At the 2018Q4 snapshot only 67.5% of 2016 loans had finished, and the finished ones are skewed toward early defaults and early payoffs. **Why start in Aug-2012?** That is when 33 credit-bureau fields begin to be reported.

### The 190 features

| Group | Columns | How they were built |
|---|---:|---|
| Numeric | 62 | Median imputation (train medians), including the engineered `credit_history_months`, `term_months`, `emp_length_years` |
| Missing-value flags | 18 | 0/1 "was missing" indicators (missing often means "never happened") |
| One-hot | 107 | grade, sub_grade, home_ownership, verification_status, purpose, addr_state (rare categories grouped) |
| Target-encoded | 3 | `zip_code`, `emp_title`, `title`: smoothed, out-of-fold default rates |

## 🤖 Modeling plan

Models are added in order of complexity, and each must beat the previous one on the validation set
([full reasoning](docs/model_strategy.pdf)):

| Step | Model | Question it answers |
|---|---|---|
| M0 | Dummy baseline | What does zero intelligence score? |
| M1 | Logistic Regression | How far does a simple, explainable model get? Any leakage? |
| M2 | Random Forest | Do non-linear effects and interactions add signal? |
| M3 | LightGBM | What is the best accuracy on this data? *(main candidate)* |
| M4 | XGBoost | Is that result robust across libraries? |
| M5 | Scorecard *(optional)* | Can it be expressed as a classic credit points scorecard? |

Every model is trained in two versions: **A** with LendingClub's own `grade`, `sub_grade`, `int_rate` and `installment`, and **B** without them.
LendingClub assigns those values *after* its own risk decision, so a real approval engine would not have them yet. **B is the deployable model**, and A is a benchmark.

**Evaluation:** ROC-AUC (main), PR-AUC, KS statistic, Brier score and calibration, and the approval-rate vs bad-rate curve, reported separately for 36- and 60-month loans. Choices are made on validation, and the test set is used once.

## 🌐 How the engine will respond *(illustrative)*

```json
{
  "probability_of_default": 0.083,
  "risk_band": "B",
  "decision": "APPROVE",
  "expected_loss": 846.60,
  "top_risk_factors": ["Credit cards fairly utilised (41.2%)", "Debt consolidation loan"],
  "top_protective_factors": ["Short 36-month term", "Income comfortably covers payments"]
}
```

The model itself returns only `probability_of_default`. The band, decision, expected loss and reasons come from the decision layer and SHAP explanations.
*Numbers are placeholders until models are trained.*

## 📁 Repository structure

```
├── run_preprocessing_v2.py        # entry point: runs preprocessing V2 end to end
├── src/preprocessing_v2/          # pipeline code, one module per stage
│   ├── restore.py  split.py  features.py  pipeline.py
│   └── export.py   audits.py  report.py   inference.py  run.py
├── tests/                         # fresh-session inference test
├── notebooks/                     # data inspection, preprocessing V1 (Steps 1-10), V2 walkthrough
├── reports/                       # EDA reports, V1 audit tables
├── docs/                          # architecture diagram, model strategy PDF (+ generators)
├── artifacts_v2/                  # metadata, audit report, feature lists (fitted .joblib not tracked)
├── data/                          # raw + interim data (not tracked)
└── final_preprocessed_data_v2/    # model-ready Parquet files (not tracked)
```

## 🚀 Getting started

```bash
git clone https://github.com/mitanshkanani/AI-Credit-Risk-Loan-Approval-Engine.git
cd AI-Credit-Risk-Loan-Approval-Engine
pip install -r requirements.txt
```

1. Download the dataset into `data/` (see [Dataset](#-dataset)).
2. Run `notebooks/accepted_preprocessing.ipynb` through **Step 10**. It writes `data/interim/accepted_quality_clean.csv`.
3. Run preprocessing V2:
   ```bash
   python run_preprocessing_v2.py   # ~2 minutes, ~16 GB RAM machine recommended
   ```
   It writes `final_preprocessed_data_v2/` and `artifacts_v2/`, and ends with `PREPROCESSING V2: COMPLETE`.
4. Score a raw application with the saved pipeline:
   ```python
   from src.preprocessing_v2 import inference
   pre = inference.load_preprocessor()
   features = inference.transform_applications([{"issue_d": "Oct-2026", "loan_amnt": 12000, "term": " 36 months"}], pre)
   print(features.shape)   # (1, 190) - missing fields are imputed
   ```

## 🛠️ Tech stack

Python · pandas · NumPy · scikit-learn · PyArrow / Parquet · joblib · ReportLab · Sweetviz.
Planned: LightGBM · XGBoost · SHAP · FastAPI.

## 👤 Author

**Mitansh Kanani** · [GitHub @mitanshkanani](https://github.com/mitanshkanani)

<div align="center"><sub>If you find this project useful, consider giving it a ⭐</sub></div>
