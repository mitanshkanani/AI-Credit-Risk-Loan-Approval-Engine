# Accepted-loan preprocessing V2 report

**Status:** COMPLETE - every required audit passed

## Split summary

| split | rows | defaults | default_rate | first_issue_month | last_issue_month |
|---|---|---|---|---|---|
| train | 388124 | 66928 | 17.24% | 2012-08 | 2014-12 |
| validation | 162745 | 33120 | 20.35% | 2015-01 | 2015-06 |
| test | 212801 | 42684 | 20.06% | 2015-07 | 2015-12 |
| stress_2016_2018 | 518744 | 116295 | 22.42% | 2016-01 | 2018-12 |
| excluded_pre_2012_08 | 65685 | 10333 | 15.73% | 2007-06 | 2012-07 |

Final feature count: **190**

## Saved artifacts

- Preprocessor: `artifacts_v2/preprocessor_v2.joblib`
- Metadata: `artifacts_v2/metadata_v2.json`
- Audit report: `artifacts_v2/audit_report_v2.json`
- Feature list: `artifacts_v2/feature_list_v2.csv`
- Removed features: `artifacts_v2/removed_features_v2.csv`
- Engineered features: `artifacts_v2/engineered_features_v2.csv`
- Single-feature AUC: `artifacts_v2/single_feature_auc_v2.csv`
- Fresh-session result: `artifacts_v2/fresh_session_test_v2.json`

## Saved datasets

- train: `final_preprocessed_data_v2/X_train.parquet`, `final_preprocessed_data_v2/y_train.parquet`
- validation: `final_preprocessed_data_v2/X_validation.parquet`, `final_preprocessed_data_v2/y_validation.parquet`
- test: `final_preprocessed_data_v2/X_test.parquet`, `final_preprocessed_data_v2/y_test.parquet`
- stress_2016_2018: `final_preprocessed_data_v2/X_stress_2016_2018.parquet`, `final_preprocessed_data_v2/y_stress_2016_2018.parquet`

## Removed features

| stage | column | reason |
|---|---|---|
| V1 Step 7 (carried over) | loan_status | Target / outcome information |
| V1 Step 7 (carried over) | target | Target / outcome information |
| V1 Step 7 (carried over) | id | Identifier, not a predictive feature |
| V1 Step 7 (carried over) | member_id | Identifier, not a predictive feature |
| V1 Step 7 (carried over) | url | Administrative LendingClub listing URL |
| V1 Step 7 (carried over) | policy_code | Constant / non-informative field |
| V1 Step 7 (carried over) | initial_list_status | Platform listing/origination attribute, not an underwriting feature |
| V1 Step 7 (carried over) | funded_amnt | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | funded_amnt_inv | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | issue_d | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | pymnt_plan | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | out_prncp | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | out_prncp_inv | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | total_pymnt | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | total_pymnt_inv | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | total_rec_prncp | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | total_rec_int | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | total_rec_late_fee | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | recoveries | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | collection_recovery_fee | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | last_pymnt_d | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | last_pymnt_amnt | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | next_pymnt_d | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | last_credit_pull_d | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | last_fico_range_high | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | last_fico_range_low | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_flag | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_type | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_reason | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_status | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | deferral_term | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_amount | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_start_date | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_end_date | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | payment_plan_start_date | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_length | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_dpd | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_loan_status | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | orig_projected_additional_accrued_interest | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_payoff_balance_amount | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | hardship_last_payment_amount | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | disbursement_method | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | debt_settlement_flag | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | debt_settlement_flag_date | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | settlement_status | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | settlement_date | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | settlement_amount | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | settlement_percentage | Post-prediction / post-origination information |
| V1 Step 7 (carried over) | settlement_term | Post-prediction / post-origination information |
| V2 by design | desc | Free-text loan description; Step 9 marked it DROP_INITIAL_VERSION (needs a separate NLP pipeline). |
| V2 replaced by engineered feature | earliest_cr_line | Replaced by `credit_history_months` |
| V2 replaced by engineered feature | sec_app_earliest_cr_line | Replaced by `sec_app_credit_history_months` |
| V2 replaced by engineered feature | term | Replaced by `term_months` |
| V2 replaced by engineered feature | emp_length | Replaced by `emp_length_years` |
| V2 key column | issue_d | Restored only for the temporal split and credit_history_months; never a model feature |
| V2 train-only filter | application_type | Constant (single value) in training rows |
| V2 train-only filter | annual_inc_joint | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | dti_joint | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | verification_status_joint | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_acc_6m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_act_il | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_il_12m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_il_24m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | mths_since_rcnt_il | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | total_bal_il | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | il_util | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_rv_12m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | open_rv_24m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | max_bal_bc | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | all_util | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | inq_fi | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | total_cu_tl | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | inq_last_12m | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | revol_bal_joint | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_fico_range_low | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_fico_range_high | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_inq_last_6mths | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_mort_acc | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_open_acc | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_revol_util | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_open_act_il | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_num_rev_accts | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_chargeoff_within_12_mths | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_collections_12_mths_ex_med | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_mths_since_last_major_derog | Missing in 100.0% of training rows (> 95%) |
| V2 train-only filter | sec_app_credit_history_months | Missing in 100.0% of training rows (> 95%) |
| V2 duplicate content (train) | missingindicator_tot_cur_bal | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_total_rev_hi_lim | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_mo_sin_rcnt_rev_tl_op | Identical to `missingindicator_mo_sin_old_rev_tl_op` in training rows |
| V2 duplicate content (train) | missingindicator_mo_sin_rcnt_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_accts_ever_120_pd | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_actv_bc_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_actv_rev_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_bc_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_il_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_op_rev_tl | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_rev_accts | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_rev_tl_bal_gt_0 | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_tl_30dpd | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_tl_90g_dpd_24m | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_num_tl_op_past_12m | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_tot_hi_cred_lim | Identical to `missingindicator_tot_coll_amt` in training rows |
| V2 duplicate content (train) | missingindicator_total_il_high_credit_limit | Identical to `missingindicator_tot_coll_amt` in training rows |

## Engineered features

| type | feature | definition | status |
|---|---|---|---|
| Row-level engineering | credit_history_months | (issue_year - ecl_year) * 12 + (issue_month - ecl_month), from earliest_cr_line | in final features |
| Row-level engineering | sec_app_credit_history_months | Same formula from sec_app_earliest_cr_line | dropped: Missing in 100.0% of training rows (> 95%) |
| Row-level engineering | term_months | Numeric months parsed from `term` (36 / 60) | in final features |
| Row-level engineering | emp_length_years | `< 1 year` -> 0 ... `10+ years` -> 10 | in final features |
| Missing-value indicator | missingindicator_mths_since_last_delinq | 1 if `mths_since_last_delinq` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_last_record | 1 if `mths_since_last_record` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_revol_util | 1 if `revol_util` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_last_major_derog | 1 if `mths_since_last_major_derog` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_tot_coll_amt | 1 if `tot_coll_amt` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_avg_cur_bal | 1 if `avg_cur_bal` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_bc_open_to_buy | 1 if `bc_open_to_buy` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_bc_util | 1 if `bc_util` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mo_sin_old_il_acct | 1 if `mo_sin_old_il_acct` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mo_sin_old_rev_tl_op | 1 if `mo_sin_old_rev_tl_op` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_recent_bc | 1 if `mths_since_recent_bc` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_recent_bc_dlq | 1 if `mths_since_recent_bc_dlq` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_recent_inq | 1 if `mths_since_recent_inq` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_mths_since_recent_revol_delinq | 1 if `mths_since_recent_revol_delinq` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_num_tl_120dpd_2m | 1 if `num_tl_120dpd_2m` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_pct_tl_nvr_dlq | 1 if `pct_tl_nvr_dlq` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_percent_bc_gt_75 | 1 if `percent_bc_gt_75` was missing (median-imputed) | in final features |
| Missing-value indicator | missingindicator_emp_length_years | 1 if `emp_length_years` was missing (median-imputed) | in final features |
| One-hot encoding | grade_* | 7 columns (rare train categories -> infrequent) | in final features |
| One-hot encoding | sub_grade_* | 33 columns (rare train categories -> infrequent) | in final features |
| One-hot encoding | home_ownership_* | 4 columns (rare train categories -> infrequent) | in final features |
| One-hot encoding | verification_status_* | 3 columns (rare train categories -> infrequent) | in final features |
| One-hot encoding | purpose_* | 13 columns (rare train categories -> infrequent) | in final features |
| One-hot encoding | addr_state_* | 47 columns (rare train categories -> infrequent) | in final features |
| Target encoding (cross-fitted) | zip_code | Smoothed training default rate of the category; out-of-fold for train rows | in final features |
| Target encoding (cross-fitted) | emp_title | Smoothed training default rate of the category; out-of-fold for train rows | in final features |
| Target encoding (cross-fitted) | title | Smoothed training default rate of the category; out-of-fold for train rows | in final features |

## Audit results

| category | check | status | detail |
|---|---|---|---|
| Target alignment | raw eligible row count == Step 10 row count | PASS | 1348099 == 1348099 |
| Target alignment | 12 shared columns identical row-by-row (raw vs Step 10) | PASS | {'loan_amnt': 0, 'int_rate': 0, 'annual_inc': 0, 'dti': 0, 'revol_util': 0, 'fico_range_low': 0, 'term': 0, 'grade': 0, 'emp_title': 0, 'title': 0, 'zip_code': 0, 'earliest_cr_line': 0} |
| Feature engineering | credit_history_months has no negative values (TRAIN) | PASS | {'missing': 0, 'negative': 0, 'min': 36.0, 'median': 177.0, 'max': 842.0} |
| Feature engineering | term parsed to {36, 60} months | PASS | [36.0, 60.0] |
| Feature engineering | every non-missing emp_length mapped to years | PASS | unmapped = 0 |
| Leakage (train-only fitting) | column filter decisions reproduce from TRAIN | PASS | 31 columns dropped |
| Leakage (train-only fitting) | every filtered column handled by a transformer (no silent remainder) | PASS | 71 consumed / 71 available |
| Leakage (train-only fitting) | numeric medians equal TRAIN medians | PASS | 62 numeric columns |
| Leakage (train-only fitting) | one-hot categories are a subset of TRAIN values | PASS | non-train categories = {} |
| Leakage (train-only fitting) | target-encoder prior equals TRAIN default rate | PASS | prior 0.172440 vs train 0.172440 |
| Leakage (train-only fitting) | target-encoder categories are a subset of TRAIN values | PASS | non-train categories = {} |
| Leakage (target encoding) | TRAIN encodings are out-of-fold (cross-fitted), not self-encoded | PASS | share of train rows whose OOF encoding differs from full-fit encoding = {'zip_code': 0.9999, 'emp_title': 0.9088, 'title': 0.9777} |
| Duplicate columns | [train] no two features have identical values | PASS | duplicate pairs = [] |
| Leakage (single-feature AUC review) | no single feature has TRAIN AUC > 0.8 (review flag, not a failure) | PASS | flagged = []; top-5 = [['int_rate', 0.6741], ['term_months', 0.6053], ['fico_range_high', 0.5836], ['fico_range_low', 0.5836], ['dti', 0.5707]] |
| Leakage (feature names) | no target / id / date / post-origination column among features | PASS | offending features = [] |
| Missing values | [train] no NaN in features | PASS | NaN count = 0 |
| Infinite values | [train] no +/-inf in features | PASS | inf count = 0 |
| Non-numeric features | [train] every feature is numeric | PASS | non-numeric = [] |
| Duplicate columns | [train] no duplicate column names | PASS | duplicates = [] |
| Feature alignment | [train] columns identical (names + order) to fitted feature list | PASS | [190] columns |
| Feature count | [train] feature count matches fitted pipeline | PASS | [190] vs 190 |
| Target alignment | [train] X rows == y rows | PASS | X 388124 / y 388124 |
| Target alignment | [train] target has no NaN and only {0, 1} | PASS | values = [0, 1] |
| Target alignment | [train] ids unique within split | PASS | 388124 unique ids |
| Target alignment | [train] exported X and y files have equal rows and y matches memory | PASS | X file rows 388124, y file rows 388124 |
| Missing values | [validation] no NaN in features | PASS | NaN count = 0 |
| Infinite values | [validation] no +/-inf in features | PASS | inf count = 0 |
| Non-numeric features | [validation] every feature is numeric | PASS | non-numeric = [] |
| Duplicate columns | [validation] no duplicate column names | PASS | duplicates = [] |
| Feature alignment | [validation] columns identical (names + order) to fitted feature list | PASS | [190] columns |
| Feature count | [validation] feature count matches fitted pipeline | PASS | [190] vs 190 |
| Target alignment | [validation] X rows == y rows | PASS | X 162745 / y 162745 |
| Target alignment | [validation] target has no NaN and only {0, 1} | PASS | values = [0, 1] |
| Target alignment | [validation] ids unique within split | PASS | 162745 unique ids |
| Target alignment | [validation] exported X and y files have equal rows and y matches memory | PASS | X file rows 162745, y file rows 162745 |
| Missing values | [test] no NaN in features | PASS | NaN count = 0 |
| Infinite values | [test] no +/-inf in features | PASS | inf count = 0 |
| Non-numeric features | [test] every feature is numeric | PASS | non-numeric = [] |
| Duplicate columns | [test] no duplicate column names | PASS | duplicates = [] |
| Feature alignment | [test] columns identical (names + order) to fitted feature list | PASS | [190] columns |
| Feature count | [test] feature count matches fitted pipeline | PASS | [190] vs 190 |
| Target alignment | [test] X rows == y rows | PASS | X 212801 / y 212801 |
| Target alignment | [test] target has no NaN and only {0, 1} | PASS | values = [0, 1] |
| Target alignment | [test] ids unique within split | PASS | 212801 unique ids |
| Target alignment | [test] exported X and y files have equal rows and y matches memory | PASS | X file rows 212801, y file rows 212801 |
| Missing values | [stress_2016_2018] no NaN in features | PASS | NaN count = 0 |
| Infinite values | [stress_2016_2018] no +/-inf in features | PASS | inf count = 0 |
| Non-numeric features | [stress_2016_2018] every feature is numeric | PASS | non-numeric = [] |
| Duplicate columns | [stress_2016_2018] no duplicate column names | PASS | duplicates = [] |
| Feature alignment | [stress_2016_2018] columns identical (names + order) to fitted feature list | PASS | [190] columns |
| Feature count | [stress_2016_2018] feature count matches fitted pipeline | PASS | [190] vs 190 |
| Target alignment | [stress_2016_2018] X rows == y rows | PASS | X 518744 / y 518744 |
| Target alignment | [stress_2016_2018] target has no NaN and only {0, 1} | PASS | values = [0, 1] |
| Target alignment | [stress_2016_2018] ids unique within split | PASS | 518744 unique ids |
| Target alignment | [stress_2016_2018] exported X and y files have equal rows and y matches memory | PASS | X file rows 518744, y file rows 518744 |
| Fitted-transformer persistence | reloaded preprocessor output identical to in-memory output | PASS | 2,000 validation rows |
| Temporal ordering | [train] every issue month inside 2012-08-01 .. 2015-01-01 (exclusive) | PASS | 2012-08 .. 2014-12 |
| Temporal ordering | [validation] every issue month inside 2015-01-01 .. 2015-07-01 (exclusive) | PASS | 2015-01 .. 2015-06 |
| Temporal ordering | [test] every issue month inside 2015-07-01 .. 2016-01-01 (exclusive) | PASS | 2015-07 .. 2015-12 |
| Temporal ordering | [stress_2016_2018] every issue month inside 2016-01-01 .. 2019-01-01 (exclusive) | PASS | 2016-01 .. 2018-12 |
| Temporal ordering | latest train month < earliest validation month | PASS | 2014-12 < 2015-01 |
| Temporal ordering | latest validation month < earliest test month | PASS | 2015-06 < 2015-07 |
| Temporal ordering | latest test month < earliest stress_2016_2018 month | PASS | 2015-12 < 2016-01 |
| Temporal ordering | no loan id appears in more than one split | PASS | 1282414 ids, 0 repeated |
| Target alignment | 1000 random ids: target matches raw loan_status | PASS | match rate = 1.0000 |
| Target alignment | 1000 random ids: issue_d matches raw | PASS | match rate = 1.0000 |
| Fitted-transformer persistence | preprocessor_v2.joblib and metadata_v2.json exist | PASS | missing = [] |
| Feature count | every exported X file has the metadata feature count and order | PASS | {'train': 190, 'validation': 190, 'test': 190, 'stress_2016_2018': 190} vs metadata 190 |
| Modeling readiness | LC risk features (grade, sub_grade, int_rate, installment) available for A/B | PASS | 42 columns |
| Fresh-session loading | separate Python process loads preprocessor and transforms raw rows + new application | PASS | exit code 0 |
| Backups untouched | temporary staging folder removed | PASS | final_preprocessed_data_v2\_staging |
| Backups untouched | final_preprocessed_data/, processed_data/, step13_artifacts/ unchanged | PASS | 8 files checked |

## Fresh-session inference test

Result: **PASSED**

| check | status | detail |
|---|---|---|
| preprocessor loads in a fresh process | PASS | C:\Users\mitansh\Desktop\ML PROJECTS\AI-Credit-Risk-Loan-Approval-Engine\artifacts_v2\preprocessor_v2.joblib |
| raw CSV rows for 5 test ids reproduce exported X_test rows exactly | PASS | ids = ['66494948', '62205382', '61963243', '59170357', '57274614']; max |diff| = 0.000e+00; columns match = True |
| new application -> 1 row with the full feature set | PASS | shape = (1, 190) |
| new application -> no NaN / inf | PASS | NaN = 0, inf = 0 |
| unseen zip_code falls back to the TRAIN prior default rate | PASS | '000xx' absent from 874 train zips = True; encoded 0.172440 vs train prior 0.172440 |
| missing emp_title uses the TRAIN-learned __MISSING__ encoding | PASS | encoded 0.213949 vs expected 0.213949 |
| missing dti is imputed with the TRAIN median | PASS | dti 17.26 vs train median 17.26; missing flag = None |
| credit_history_months computed from application month | PASS | Mar-2010 -> Oct-2026 = 199.0 months |
