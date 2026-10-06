---
title: AI Credit Risk Engine
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
short_description: Loan approval engine trained on 388k LendingClub loans
---

# AI Credit Risk & Loan Approval Engine

Enter a loan application and get a calibrated **probability of default**, a **risk band** (R1–R7), an
**approve / review / decline** decision, the **expected loss** in dollars, and the **main reasons**
(SHAP reason codes).

- Model: LightGBM with monotonic constraints, 146 features, trained on LendingClub accepted loans 2012–2014.
- Sealed 2015 H2 test set (212,801 loans, opened once): ROC-AUC 0.746 vs 0.715 for LendingClub's own interest rate.
- API: `POST /api/score` (see `/docs`).
- Code and full write-up: https://github.com/mitanshkanani/AI-Credit-Risk-Loan-Approval-Engine

Educational project, not a real credit decision.
