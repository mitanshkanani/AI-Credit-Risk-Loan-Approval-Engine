"""Generate docs/preprocessing_architecture.svg.

    python docs/build_preprocessing_diagram.py

The diagram is meant to be read top to bottom: every box is one step of the
accepted-loan preprocessing, with what was done, why, the real numbers, and
the files it produced.
"""

import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).with_name("preprocessing_architecture.svg")

WIDTH = 1240
MARGIN = 40
BOX_W = WIDTH - 2 * MARGIN
PAD = 18
LINE = 19
WRAP = 162
GAP = 34

PHASES = {
    "v1": {"fill": "#eef4ff", "stroke": "#3b6fd4", "badge": "#3b6fd4"},
    "warn": {"fill": "#fff4e5", "stroke": "#d4840f", "badge": "#d4840f"},
    "v2": {"fill": "#ecf8f0", "stroke": "#2e9b5b", "badge": "#2e9b5b"},
    "audit": {"fill": "#f4effd", "stroke": "#7a4fd1", "badge": "#7a4fd1"},
    "next": {"fill": "#f3f4f6", "stroke": "#4b5563", "badge": "#4b5563"},
}

# Each step: (phase, badge, title, file tag, items)
# item kinds: ("p", text) paragraph, ("b", text) bullet, ("o", text) output line,
#             ("t", [[cells], ...]) table, ("h", text) sub-heading
STEPS = [
    ("v1", "1", "Load the raw data and define the prediction point", "accepted_preprocessing.ipynb · Step 1", [
        ("p", "Loaded LendingClub accepted_2007_to_2018Q4.csv: 2,260,701 loans × 151 columns."),
        ("p", "Ground rule for the whole project: the model scores a NEW application, so it may only use information "
              "that exists at or before the application / underwriting decision. Every later step enforces this rule."),
    ]),
    ("v1", "2", "Investigate loan_status (what happened to each loan)", "Step 2", [
        ("p", "9 distinct statuses + 33 missing values:"),
        ("t", [["Fully Paid", "1,076,751"], ["Current (still running)", "878,317"], ["Charged Off", "268,559"],
               ["Late (31-120 / 16-30 days) + In Grace Period", "34,252"],
               ["'Does not meet the credit policy' (Fully Paid / Charged Off)", "1,988 / 761"], ["Default", "40"]]),
    ]),
    ("v1", "3", "Define the target (what the model predicts)", "Step 3", [
        ("b", "target = 0 (good): Fully Paid, 'Does not meet the credit policy. Status: Fully Paid'."),
        ("b", "target = 1 (bad / default): Charged Off, Default, 'Does not meet the credit policy. Status: Charged Off'."),
        ("b", "No target: Current, Late, In Grace Period, missing. These loans have no final outcome yet, so "
              "labelling them would be guessing."),
    ]),
    ("v1", "4", "Define the eligible population (only loans with a final outcome)", "Step 4", [
        ("p", "Kept 1,348,099 resolved loans (59.63% of all rows) and dropped 912,602 unresolved ones."),
        ("p", "Class balance: 1,078,739 good (80.0%) vs 269,360 default (20.0%)."),
    ]),
    ("v1", "5", "Audit every column", "Step 5", [
        ("p", "For all 152 columns (151 + target): data type, missing count and %, number of unique values, sample values."),
        ("p", "Found 40 columns ≥ 80% missing and 7 constant / single-value columns."),
        ("o", "accepted_column_audit.csv"),
    ]),
    ("v1", "6", "Application-time vs future-time audit (the leakage review)", "Step 6", [
        ("p", "Every column was put in exactly one class, by asking: would we know this value when the application arrives?"),
        ("b", "94 application-time features: income, DTI, FICO, credit-bureau history, loan amount, purpose, state, ..."),
        ("b", "42 post-prediction columns, only known AFTER the loan starts: payments received, recoveries, outstanding "
              "principal, last payment / last FICO / last credit pull, hardship plans, debt settlement, funded amounts, issue_d."),
        ("b", "4 LendingClub-derived columns kept for review: int_rate, installment, grade, sub_grade (LendingClub's own risk view)."),
        ("b", "Other: 4 text / high-cardinality (emp_title, desc, title, zip_code), 1 historical date (earliest_cr_line), "
              "2 target (loan_status, target), 2 identifiers (id, member_id), 3 administrative (url, policy_code, initial_list_status)."),
        ("o", "accepted_timing_audit.csv"),
    ]),
    ("v1", "7", "Remove target, leakage, identifiers and administrative columns", "Step 7", [
        ("p", "Dropped 49 columns = 2 target + 2 identifiers + 3 administrative + 42 post-prediction. "
              "Result: 1,348,099 rows × 103 candidate columns. Checks confirmed no target, id or future column remains."),
        ("o", "accepted_step7_removal_audit.csv, accepted_model_candidates.csv"),
    ]),
    ("v1", "8", "Data-quality investigation", "Step 8", [
        ("p", "Checked missingness, all-missing columns, constant columns, duplicate column names, columns with identical "
              "content, duplicate rows, infinite values, numeric ranges and categorical cardinality. Nothing needed "
              "removing, so the dataset stays 1,348,099 × 103."),
        ("o", "accepted_step8_quality_audit.csv, accepted_quality_clean.csv  (the starting point for V2)"),
    ]),
    ("v1", "9", "Feature semantics and representation plan", "Step 9", [
        ("p", "Each column labelled numeric / categorical / date / free text / LendingClub-derived, with a planned "
              "treatment. Key decisions: desc (free text) is left out of the first version because it needs NLP; "
              "earliest_cr_line becomes a date-based feature; emp_title / title need a high-cardinality encoder."),
        ("o", "accepted_step9_representation_plan.csv"),
    ]),
    ("v1", "10", "Missing-value strategy", "Step 10", [
        ("p", "Columns bucketed by missingness, with a strategy for each. Rule carried into V2: every imputation value "
              "(median, category, ...) must be learned from TRAINING data only, never from validation or test."),
        ("o", "accepted_step10_missing_audit.csv, accepted_step10_missing_strategy.csv"),
    ]),
    ("warn", "!", "Why V1 Steps 11-16 were replaced by Preprocessing V2", "review outcome", [
        ("p", "V1 finished with a random 80/10/10 split and 1,943 columns, but a review found five problems:"),
        ("b", "the labels (y) were never exported, so a modeling notebook could not load them;"),
        ("b", "earliest_cr_line (a date like 'Aug-2003') was one-hot encoded into 733 columns instead of one number;"),
        ("b", "zip_code was one-hot encoded into 936 columns;"),
        ("b", "the final fitted encoders and medians were never saved, so a new application could not be scored;"),
        ("b", "the random split mixes years; credit-risk models should be tested on LATER loans than they were trained on."),
        ("p", "V1 outputs are kept untouched as a backup: final_preprocessed_data/, processed_data/, step13_artifacts/."),
    ]),
    ("v2", "V2·1", "Data restoration: re-attach id, issue_d and target safely", "src/preprocessing_v2/restore.py", [
        ("p", "accepted_quality_clean.csv has no id, issue_d or target (Step 7 removed them). The raw CSV is re-read in "
              "its original order and filtered with the same target mapping: 1,348,099 eligible rows."),
        ("p", "Safety check before attaching anything by position: 12 shared columns (loan_amnt, int_rate, annual_inc, dti, "
              "revol_util, fico_range_low, term, grade, emp_title, title, zip_code, earliest_cr_line) are compared row by row. "
              "Result: 0 mismatches, ids unique, every issue_d parses. Any mismatch would stop the run."),
    ]),
    ("v2", "V2·2", "Temporal split by loan issue month", "split.py", [
        ("p", "Train on older loans, validate and test on newer ones, the way the model will actually be used:"),
        ("t", [["Split", "Issue months", "Rows", "Defaults", "Default rate"],
               ["Train", "2012-08 → 2014-12", "388,124", "66,928", "17.24%"],
               ["Validation", "2015-01 → 2015-06", "162,745", "33,120", "20.35%"],
               ["Test", "2015-07 → 2015-12", "212,801", "42,684", "20.06%"],
               ["Stress test only", "2016-01 → 2018-12", "518,744", "116,295", "22.42%"],
               ["Excluded", "2007-06 → 2012-07", "65,685", "10,333", "15.73%"]]),
        ("b", "Why train starts in Aug-2012: 33 credit-bureau fields (tot_cur_bal, num_*, mo_sin_*, ...) only start being "
              "filled in Mar-Aug 2012. Earlier loans would teach the model that 'missing' means 'old loan'."),
        ("b", "Why 2016-2018 is not the test set: at the 2018Q4 snapshot only 67.5% of 2016 loans (38% of 2017, 11% of 2018) "
              "had finished. The finished ones are mostly early defaults and early payoffs, so their default rate is distorted."),
        ("b", "Every split has plenty of defaults (≥ 33k), so no resampling is needed."),
    ]),
    ("v2", "V2·3", "Feature engineering (row by row, nothing learned from data)", "features.py", [
        ("b", "credit_history_months = (issue_year − ecl_year) × 12 + (issue_month − ecl_month), using the application month. "
              "Replaces the 733 date columns with ONE number. Train: no missing, no negatives, range 36-842, median 177."),
        ("b", "term_months: ' 36 months' → 36, ' 60 months' → 60.    emp_length_years: '< 1 year' → 0 ... '10+ years' → 10."),
        ("b", "Text categories stripped; zip_code / emp_title / title lowercased so 'Teacher' and 'teacher ' are the same value."),
        ("b", "All other columns converted to numbers; raw dates, id, issue_d and desc dropped."),
        ("p", "This exact function lives INSIDE the saved pipeline, so a new application goes through the same steps."),
    ]),
    ("v2", "V2·4", "Fitted preprocessing: one scikit-learn Pipeline, fitted on TRAIN only", "pipeline.py", [
        ("p", "fit_transform(X_train, y_train) is the only fit. Validation, test and stress rows are only ever transformed."),
        ("h", "a) Train-only column filter: drop columns > 95% missing or constant in TRAIN (31 dropped)"),
        ("b", "application_type (only 'Individual' in train); 4 joint-application + 11 second-applicant columns + "
              "sec_app_credit_history_months (joint applications started later);"),
        ("b", "14 newer credit-bureau columns (open_acc_6m, il_util, all_util, inq_fi, ...) that LendingClub only began reporting in late 2015."),
        ("h", "b) ColumnTransformer: three branches"),
        ("b", "Numeric (62 columns): fill missing values with the TRAIN median, plus a 0/1 'was missing' flag "
              "(missing often means 'never happened', e.g. months since last delinquency)."),
        ("b", "One-hot (6 columns): grade 7, sub_grade 33, home_ownership 4, verification_status 3, purpose 13, addr_state 47. "
              "Categories with < 500 training rows are grouped as 'infrequent'; unseen ones map safely."),
        ("b", "Target encoding (zip_code, emp_title, title): each category becomes its smoothed TRAIN default rate. Training rows "
              "use 5-fold out-of-fold values (a row never sees its own label); val/test use the train-only mapping; "
              "unseen categories get the overall train default rate (17.24%)."),
        ("h", "c) Drop duplicate columns: 17 'was missing' flags were identical to another flag in TRAIN"),
        ("p", "Result: 103 raw input columns → 190 numeric model features."),
    ]),
    ("v2", "V2·5", "Artifact saving", "export.py", [
        ("b", "preprocessor_v2.joblib: the whole fitted pipeline in one file (feature engineering + filter + encoders + dedupe)."),
        ("b", "metadata_v2.json: split windows and statistics, input columns, the 190 feature names, feature groups "
              "(incl. the 42 LendingClub risk columns for the with/without comparison), parameters, library versions."),
        ("o", "artifacts_v2/preprocessor_v2.joblib, metadata_v2.json, feature_list_v2.csv"),
    ]),
    ("v2", "V2·6", "Export every split", "export.py", [
        ("p", "Each split is transformed (validation / test / stress in 100k-row chunks to save memory), written, then read back and checked."),
        ("b", "X_<split>.parquet: the 190 features only."),
        ("b", "y_<split>.parquet: id, issue_d, target, in exactly the same row order as X."),
        ("o", "final_preprocessed_data_v2/  (train, validation, test, stress_2016_2018)"),
    ]),
    ("audit", "V2·7", "Final audits: 72 checks, 0 failed, 0 warnings", "audits.py", [
        ("b", "Target alignment (20): raw vs Step 10 rows identical; X rows = y rows; target only 0/1; "
              "1,000 random ids match the raw loan_status and issue_d 100%."),
        ("b", "Temporal ordering (8): every row inside its window; 2014-12 < 2015-01, 2015-06 < 2015-07, "
              "2015-12 < 2016-01; no id in two splits."),
        ("b", "Leakage (9): no target / id / date / post-loan column among features; medians, categories, filter "
              "decisions and the encoder prior all reproduce from TRAIN alone; training-row target encodings are out-of-fold. "
              "Highest single-feature AUC = 0.674 (int_rate), under the 0.80 review flag."),
        ("b", "Data hygiene (17): 0 NaN, 0 infinite, 0 non-numeric, 0 duplicate columns in every split."),
        ("b", "Consistency (9): 190 features in identical order in every file and in the metadata."),
        ("b", "Persistence and backups (4): reloaded pipeline gives identical output; V1 backup files unchanged."),
        ("b", "Engineering, readiness, fresh session (5): no negative credit history; term ∈ {36, 60}; every emp_length mapped; "
              "LendingClub risk columns available; fresh-process test passed."),
        ("o", "artifacts_v2/audit_report_v2.json, preprocessing_v2_report.md, single_feature_auc_v2.csv"),
    ]),
    ("audit", "✓", "Fresh-session inference test (separate Python process)", "tests/fresh_session_inference_test.py", [
        ("b", "Loads preprocessor_v2.joblib with no notebook state."),
        ("b", "Raw CSV rows for 5 test loans reproduce the exported X_test rows exactly (max difference 0.0)."),
        ("b", "A hand-written NEW application gives 1 row × 190 features with no NaN: unseen zip '000xx' → train default rate "
              "0.1724; missing emp_title → its learned 'missing' value; missing dti → train median 17.26; "
              "Mar-2010 → Oct-2026 = 199 months of credit history."),
    ]),
    ("next", "→", "Next: model preparation (not started)", "", [
        ("b", "Model A uses all 190 features; model B drops the 42 LendingClub risk columns (grade, sub_grade, int_rate, "
              "installment) to show what the model learns on its own."),
        ("b", "Logistic Regression baseline (needs a scaler) → LightGBM / XGBoost; class weights for the 17-20% default rate."),
        ("b", "Tune on validation, test once on test, calibrate probabilities, report results for 36- vs 60-month loans "
              "separately, optionally check on the 2016-2018 stress set."),
    ]),
]


def wrap_items(items):
    """Turn items into renderable lines: (kind, text_or_cells, indent)."""
    lines = []
    for kind, content in items:
        if kind == "t":
            for i, row in enumerate(content):
                lines.append(("trow_head" if i == 0 and len(content[0]) == 5 else "trow", row, 0))
        elif kind == "b":
            wrapped = textwrap.wrap(content, WRAP - 4)
            for i, part in enumerate(wrapped):
                lines.append(("bullet" if i == 0 else "cont", part, 18))
        elif kind == "h":
            lines.append(("head", content, 0))
        elif kind == "o":
            lines.append(("out", "Output: " + content, 0))
        else:
            for part in textwrap.wrap(content, WRAP):
                lines.append(("para", part, 0))
    return lines


def table_columns(row):
    if len(row) == 5:
        return [0, 170, 360, 470, 580]
    return [0, 560]


def render():
    parts = []
    y = 40
    title_block = [
        f'<text x="{MARGIN}" y="{y + 24}" class="title">Accepted-loan preprocessing: architecture</text>',
        f'<text x="{MARGIN}" y="{y + 50}" class="sub">AI Credit Risk / Loan Approval Engine · read top to bottom · '
        'blue = original notebook (Steps 1-10, kept) · orange = review · green = Preprocessing V2 · purple = audits</text>',
    ]
    parts.extend(title_block)
    y += 80

    centers = []
    for phase, badge, title, tag, items in STEPS:
        colors = PHASES[phase]
        lines = wrap_items(items)
        height = PAD + 30 + len(lines) * LINE + PAD
        parts.append(
            f'<rect x="{MARGIN}" y="{y}" width="{BOX_W}" height="{height}" rx="10" '
            f'fill="{colors["fill"]}" stroke="{colors["stroke"]}" stroke-width="1.5"/>'
        )
        badge_w = 22 + 9 * len(badge)
        parts.append(
            f'<rect x="{MARGIN + PAD}" y="{y + PAD - 2}" width="{badge_w}" height="24" rx="12" fill="{colors["badge"]}"/>'
            f'<text x="{MARGIN + PAD + badge_w / 2}" y="{y + PAD + 15}" class="badge" text-anchor="middle">{escape(badge)}</text>'
        )
        parts.append(
            f'<text x="{MARGIN + PAD + badge_w + 12}" y="{y + PAD + 16}" class="step">{escape(title)}</text>'
        )
        if tag:
            parts.append(
                f'<text x="{MARGIN + BOX_W - PAD}" y="{y + PAD + 15}" class="tag" text-anchor="end">{escape(tag)}</text>'
            )
        ty = y + PAD + 30 + 14
        for kind, content, indent in lines:
            x = MARGIN + PAD + indent
            if kind in ("trow", "trow_head"):
                cls = "cellh" if kind == "trow_head" else "cell"
                for cx, cell in zip(table_columns(content), content):
                    parts.append(f'<text x="{x + 20 + cx}" y="{ty}" class="{cls}">{escape(cell)}</text>')
            elif kind == "bullet":
                parts.append(f'<circle cx="{x - 9}" cy="{ty - 4.5}" r="2.6" fill="{colors["stroke"]}"/>')
                parts.append(f'<text x="{x}" y="{ty}" class="body">{escape(content)}</text>')
            elif kind == "head":
                parts.append(f'<text x="{x}" y="{ty}" class="head" fill="{colors["stroke"]}">{escape(content)}</text>')
            elif kind == "out":
                parts.append(f'<text x="{x}" y="{ty}" class="out">{escape(content)}</text>')
            else:
                parts.append(f'<text x="{x}" y="{ty}" class="body">{escape(content)}</text>')
            ty += LINE
        centers.append((y, y + height))
        y += height + GAP

    # arrows between consecutive boxes
    for (top1, bottom1), (top2, _) in zip(centers, centers[1:]):
        cx = WIDTH / 2
        parts.append(
            f'<line x1="{cx}" y1="{bottom1 + 3}" x2="{cx}" y2="{top2 - 5}" stroke="#6b7280" '
            f'stroke-width="2" marker-end="url(#arrow)"/>'
        )

    height = y + 10
    style = """
    <style>
      text { font-family: 'Segoe UI', Helvetica, Arial, sans-serif; fill: #1f2937; }
      .title { font-size: 26px; font-weight: 700; }
      .sub { font-size: 13.5px; fill: #4b5563; }
      .step { font-size: 17px; font-weight: 700; }
      .badge { font-size: 13px; font-weight: 700; fill: #ffffff; }
      .tag { font-size: 12.5px; fill: #6b7280; font-family: Consolas, 'Courier New', monospace; }
      .body { font-size: 13.5px; }
      .head { font-size: 13.5px; font-weight: 700; }
      .out { font-size: 12.5px; fill: #374151; font-family: Consolas, 'Courier New', monospace; }
      .cell { font-size: 13px; font-family: Consolas, 'Courier New', monospace; }
      .cellh { font-size: 13px; font-weight: 700; font-family: Consolas, 'Courier New', monospace; }
    </style>"""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}">'
        f'<title>Accepted-loan preprocessing architecture</title>{style}'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#6b7280"/></marker></defs>'
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>'
        + "".join(parts)
        + "</svg>"
    )
    OUT.write_text(svg, encoding="utf-8")
    return OUT, height


if __name__ == "__main__":
    path, h = render()
    print(f"wrote {path} ({WIDTH} x {h})")
