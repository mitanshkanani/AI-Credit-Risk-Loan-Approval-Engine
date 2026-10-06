"""Build docs/guide/AI_Credit_Risk_Project_Guide.pdf: the complete study guide for this project.

    python docs/guide/make_figures.py      (once, makes docs/guide/figures/)
    python docs/guide/build_guide.py
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
OUT = HERE / "AI_Credit_Risk_Project_Guide.pdf"

F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Body", F + "calibri.ttf"))
pdfmetrics.registerFont(TTFont("Body-B", F + "calibrib.ttf"))
pdfmetrics.registerFont(TTFont("Body-I", F + "calibrii.ttf"))
pdfmetrics.registerFont(TTFont("Body-BI", F + "calibriz.ttf"))
pdfmetrics.registerFont(TTFont("Mono", F + "consola.ttf"))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-B", italic="Body-I", boldItalic="Body-BI")

INK, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
BLUE, GREEN, ORANGE, RED, PURPLE = (colors.HexColor(c) for c in ("#3b6fd4", "#2e9b5b", "#d4840f", "#c2412d", "#7c4dcc"))
S = {
    "cover": ParagraphStyle("cv", fontName="Body-B", fontSize=24, leading=28, textColor=INK),
    "coversub": ParagraphStyle("cs", fontName="Body", fontSize=12, leading=16, textColor=MUTED, spaceAfter=10),
    "h1": ParagraphStyle("h1", fontName="Body-B", fontSize=20, leading=24, textColor=INK, spaceAfter=8),
    "h2": ParagraphStyle("h2", fontName="Body-B", fontSize=14, leading=18, textColor=BLUE, spaceBefore=9, spaceAfter=4),
    "h3": ParagraphStyle("h3", fontName="Body-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=6, spaceAfter=2),
    "b": ParagraphStyle("b", fontName="Body", fontSize=10.5, leading=14.6, textColor=INK, spaceAfter=5),
    "li": ParagraphStyle("li", fontName="Body", fontSize=10.5, leading=14.2, textColor=INK, leftIndent=14, bulletIndent=3,
                         spaceAfter=2.5),
    "c": ParagraphStyle("c", fontName="Body", fontSize=9.4, leading=12, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-B", fontSize=9.4, leading=12, textColor=colors.white),
    "small": ParagraphStyle("sm", fontName="Body-I", fontSize=9, leading=12, textColor=MUTED, spaceAfter=4),
    "formula": ParagraphStyle("f", fontName="Body", fontSize=11.5, leading=17, textColor=INK, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=5),
    "mono": ParagraphStyle("m", fontName="Mono", fontSize=8.4, leading=11, textColor=INK),
    "q": ParagraphStyle("q", fontName="Body-B", fontSize=10.5, leading=14, textColor=PURPLE, spaceBefore=6, spaceAfter=1),
}
W = A4[0] - 36 * mm


def NewPage():
    return CondPageBreak(250 * mm)   # break unless already at the top of a page (no blank pages)


def p(text, style="b"):
    return Paragraph(text, S[style])


def ps(*texts):
    return [p(t) for t in texts]


def bullets(items):
    return [Paragraph(t, S["li"], bulletText="•") for t in items]


def table(rows, widths, color=BLUE):
    data = [[Paragraph(str(c), S["cw" if r == 0 else "c"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [("BACKGROUND", (0, 0), (-1, 0), color), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ca3af")),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f9fafb")) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def box(flowables, color=BLUE, fill="#eef3fc"):
    t = Table([[flowables]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)), ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    return t


def key(text):
    return box([p("<b>Remember:</b> " + text)], GREEN, "#ecf8f0")


def img(path, width=W, max_h=None):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    h = width * ih / iw
    if max_h and h > max_h:
        width, h = max_h * iw / ih, max_h
    return Image(str(path), width=width, height=h)


def formula(text):
    return p(text, "formula")


def step(n, title, paras, extra=None):
    out = [p(f"Step {n}. {title}", "h3")] + ps(*paras)
    return out + (extra or [])


R = ROOT / "models"
story = []

# =====================================================================================  PAGE 1: problem statement
story += [
    p("AI Credit Risk &amp; Loan Approval Engine", "cover"),
    p("Complete project guide: what was built, how, why, and how to explain it in an interview", "coversub"),
    p("1. Problem statement", "h1"),
    *ps(
        "A lender receives a loan application and must decide, at that moment, whether to lend. If it says yes to someone who "
        "later stops paying, it loses most of the money still owed. If it says no to someone who would have repaid, it loses "
        "the interest it would have earned and a customer. The decision has to be made with only the information available on "
        "the application day: income, debts, credit-bureau history, the loan amount and term. Nothing that happens later may be used.",
        "<b>The goal of this project</b> is to build that decision engine end to end on real data: <b>LendingClub</b>, a US "
        "peer-to-peer lender, published every loan it issued between 2007 and 2018 (2,260,701 loans, 151 columns). For each "
        "loan we know what was known at application time and what eventually happened (repaid or charged off).",
    ),
    p("What the engine must output for every application", "h2"),
    *bullets([
        "<b>Probability of default (PD)</b>: a calibrated number between 0 and 1. \"PD = 20%\" must mean that about 20 of 100 "
        "such applicants really default.",
        "<b>Risk band</b> R1 (safest) to R7: a short label, like a credit grade.",
        "<b>Decision</b>: APPROVE, REVIEW (a person looks at it) or DECLINE, plus REFER when the applicant is outside what the "
        "model has ever seen.",
        "<b>Expected loss in dollars</b>: PD × amount owed at default × share not recovered.",
        "<b>Reasons</b>: the main factors that raised or lowered this applicant's risk, in plain language (needed for "
        "adverse-action notices in US lending).",
    ]),
    p("Constraints that make it a real-world problem, not a Kaggle exercise", "h2"),
    *bullets([
        "<b>No leakage:</b> 42 of the 151 columns are only known after the loan starts (payments, recoveries, last FICO...). "
        "They must be removed or the model would look brilliant and be useless.",
        "<b>Out-of-time testing:</b> the model is trained on older loans and tested on newer ones, as it would be used.",
        "<b>No lender shortcuts:</b> LendingClub's own grade and interest rate are its risk opinion; the deployable model "
        "must not use them.",
        "<b>Imbalanced target:</b> only ~20% of loans default, so accuracy is a misleading metric.",
        "<b>Deployable:</b> one reproducible package, a public web app and an API, with input checks and guardrails.",
    ]),
    box([p("<b>Final result in one line:</b> a monotonic, calibrated LightGBM model on 146 application-time features "
           "reaches <b>ROC-AUC 0.746</b> on a sealed test set of <b>212,801</b> later loans (LendingClub's own interest rate: "
           "0.715), approves 66.8% of applicants at an 11.6% default rate (vs 20.1% overall), and runs live at "
           "<b>credit-risk-engine-dhib.onrender.com</b>.")]),
    NewPage(),
]

# =====================================================================================  PAGE 2: why this project
story += [
    p("2. Why I chose this project", "h1"),
    *ps(
        "<b>Credit risk is one of the oldest and most valuable uses of machine learning.</b> Every bank, card issuer, fintech and "
        "buy-now-pay-later company runs a probability-of-default model. It is a problem where a small improvement in ranking "
        "is worth millions, and where mistakes are expensive and regulated. Working on it shows the skills companies actually "
        "pay for: careful data work, honest evaluation, explainability and deployment, not just fitting a model.",
        "<b>The data is real and messy.</b> LendingClub's file has 2.26 million rows and 151 columns, with missing values that "
        "mean different things, columns that leak the future, free-text fields, dates stored as text and a target that has to "
        "be defined from nine different loan statuses. That makes it a much better learning project than a clean toy dataset, "
        "because most of the real work in industry is exactly this kind of cleaning and auditing.",
        "<b>It covers the whole lifecycle in one project.</b> Data understanding → leakage audit → preprocessing pipeline → "
        "model comparison from a dummy baseline up to gradient boosting → calibration → explainability (SHAP) → fairness check → "
        "business decision layer → packaging → final test → web deployment → responding to an external audit. Very few portfolio "
        "projects go past \"I trained XGBoost and got an AUC\".",
        "<b>It has a clear business answer.</b> At the end the question is not \"what is the accuracy\" but \"how many people do we "
        "approve, what default rate do we accept, and how much money do we lose\". That connects machine learning to decisions, "
        "which is what interviewers in fintech, banking and analytics roles want to hear about.",
        "<b>It is explainable by design.</b> In lending you must tell a declined applicant why. That forces the use of monotonic "
        "constraints (higher FICO can never increase risk) and SHAP reason codes, which are good topics to discuss in interviews "
        "because they show you understand that a model is used by people, not only scored by metrics.",
    ),
    key("credit risk = high business value + messy real data + full ML lifecycle + explainability. That is why it is a strong "
        "resume project."),
    NewPage(),
]

# =====================================================================================  PAGE 3: what it teaches
story += [
    p("3. What this project teaches", "h1"),
    table([
        ["Skill", "Where it appears in the project"],
        ["Problem framing", "Defining the target from 9 loan statuses; deciding which loans have a final outcome; choosing what the "
                            "model may know at decision time."],
        ["Data leakage", "Column-by-column audit: 94 application-time vs 42 post-loan columns; the single-feature AUC check."],
        ["Reproducible preprocessing", "One scikit-learn Pipeline fitted on train only, saved with joblib, tested in a fresh process."],
        ["Time-based validation", "Train 2012-08..2014-12, validate 2015 H1, test 2015 H2; why random splits overstate quality."],
        ["Encoding categorical data", "One-hot with rare-category grouping; cross-fitted target encoding for zip / job title."],
        ["Missing values", "Median imputation + 'was missing' flags, because missing often means 'never happened'."],
        ["Model selection", "A ladder M0 → M4 where each model must beat the previous one; random search with early stopping."],
        ["Imbalanced classification metrics", "ROC-AUC, PR-AUC, KS, Brier, log loss, calibration gap, ECE, and why not accuracy."],
        ["Probability calibration", "Platt scaling vs isotonic regression, chosen by out-of-fold Brier score."],
        ["Domain constraints", "Monotonic constraints in LightGBM; verified with a sweep (3,709 violations → 0)."],
        ["Explainability", "SHAP values from LightGBM, grouped into concepts, turned into plain-language reason codes."],
        ["Fairness", "Zip-code ablation (proxy for where people live); state-level calibration check."],
        ["Business decisions", "Risk bands, approve / review / decline thresholds from business targets, expected loss PD×EAD×LGD."],
        ["Honest evaluation", "Test set sealed in code, criteria committed to git before opening it, opened exactly once."],
        ["MLOps basics", "Checksummed model bundle, version pinning, fresh-session tests, FastAPI, Docker, Render, auto-deploy."],
        ["Working with constraints", "16 GB laptop: memory-safe chunked reads; heavy training moved to free Kaggle CPU kernels."],
        ["Handling critique", "Two independent black-box audits → guardrails v1.1-v1.3 without retraining the frozen model."],
    ], [48 * mm, W - 48 * mm]),
    Spacer(1, 6),
    key("the biggest lesson is that the model is the smallest part. Defining the target, removing leakage, testing honestly and "
        "turning a probability into a safe decision is most of the work."),
    NewPage(),
]

# =====================================================================================  PAGE 4: architecture
story += [
    p("4. Architecture: how the whole system works", "h1"),
    img(FIG / "architecture.png", W, max_h=225 * mm),
    NewPage(),
    p("Reading the architecture", "h2"),
    *ps(
        "<b>Left side (offline, done once).</b> The raw CSV is cleaned and audited in the V1 notebook (Steps 1-10), then the V2 "
        "code builds a single fitted preprocessing pipeline and the model-ready Parquet files. Five model families are trained "
        "and compared (heavy ones on Kaggle). The winner is finalised in stages: monotonic constraints and calibration, "
        "reason codes and fairness, the decision policy, packaging into a bundle, and the one-time test.",
        "<b>The bundle is the bridge.</b> Four files (fitted preprocessor, LightGBM model, Platt calibrator, decision policy) and "
        "a manifest with their SHA-256 checksums and library versions. The web app loads exactly these files; if a file is "
        "changed or scikit-learn is the wrong version, the engine refuses to start. That guarantees the live app scores exactly "
        "like the model that was tested.",
        "<b>Right side (online, every request).</b> A browser or API client sends JSON to Render. Render runs our Docker "
        "container: uvicorn (the web server) passes the request to the FastAPI app, which validates the input, applies the "
        "input guardrails, and calls <i>CreditRiskEngine.score()</i>. The engine runs the same preprocessing pipeline, gets the "
        "raw PD from LightGBM, calibrates it, applies the policy (band, decision, expected loss), computes SHAP reasons, then "
        "applies the output guardrails and returns JSON that the web page shows as a result card.",
    ),
    p("One request, step by step", "h3"),
    *bullets([
        "Form sends <i>{loan_amnt: 12000, term: 36, annual_inc: 92500, dti: 18.4, fico_range_low: 665, ...}</i>.",
        "Validation: required fields present, categories known, numbers valid, dates not in the future.",
        "Scope check: FICO ≥ 660, DTI ≤ 40, loan $1k-$35k and ≤ 50% of income, etc. If not → REFER, no score shown.",
        "Preprocessor: credit_history_months from the credit-line date, medians for missing fields, one-hot and target encoding "
        "→ 146 numbers in the exact training order.",
        "LightGBM (1,931 trees) → raw PD → Platt calibration → calibrated PD, e.g. 19.5%.",
        "Policy: 19.5% ≤ 22.5% → APPROVE, band R4; expected loss = 0.195 × 12,000 × 0.573 × 0.887 ≈ $1,190.",
        "SHAP: top 4 risk reasons and 2 strengths; JSON returned in ~0.7 s on the free server.",
    ]),
    NewPage(),
]

# =====================================================================================  PREPROCESSING
story += [
    p("5. Preprocessing (in detail)", "h1"),
    *ps(
        "Preprocessing turned 2.26 million raw rows with 151 columns into four clean, model-ready datasets with 190 numeric "
        "features, plus one saved pipeline that can transform a brand-new application in exactly the same way. It happened in "
        "two parts: the <b>V1 notebook</b> (Steps 1-10: understanding, target, leakage audit, quality) and <b>Preprocessing V2</b> "
        "(Python modules in <i>src/preprocessing_v2/</i>: restore keys, temporal split, feature engineering, a fitted pipeline, "
        "export, 72 audits and a fresh-session test). V2 replaced V1's later steps after a review found five problems.",
    ),
    img(FIG / "preprocessing_flow.png", W, max_h=150 * mm),
    NewPage(),
    p("Part A: understanding the data (V1 notebook, Steps 1-10)", "h2"),
    *step(1, "Load the data and define the prediction point", [
        "The raw file <i>accepted_2007_to_2018Q4.csv</i> has 2,260,701 loans and 151 columns. Before touching any column we fixed "
        "the ground rule for the whole project: the model scores a <b>new application</b>, so it may only use information that "
        "exists at or before the moment the lender decides. This single rule drives every later choice, especially the leakage audit.",
        "Why it matters: a model trained with information from the future (for example the total amount repaid) will score "
        "almost perfectly in a notebook and fail completely in real use, because that information does not exist when the "
        "application arrives. Fixing the prediction point first avoids that trap.",
    ]),
    *step(2, "Investigate loan_status", [
        "The column <i>loan_status</i> says what happened to each loan. It has 9 distinct values plus 33 missing ones: Fully Paid, "
        "Charged Off, Current, Late (31-120 days), Late (16-30 days), In Grace Period, Default, and two 'Does not meet the "
        "credit policy' variants (Fully Paid / Charged Off) for older loans issued under a previous policy.",
        "Understanding these values is necessary to define the target correctly. 'Current' or 'Late' loans have no final outcome "
        "yet: a late loan may still recover and a current loan may still default. Treating them as good or bad would be guessing.",
    ]),
    *step(3, "Define the target", [
        "<b>target = 1 (default)</b>: Charged Off, Default, and 'Does not meet the credit policy: Charged Off'. "
        "<b>target = 0 (repaid)</b>: Fully Paid and 'Does not meet the credit policy: Fully Paid'. Current, Late, In Grace Period "
        "and missing statuses get no target.",
        "Charged Off means LendingClub gave up collecting (usually after 120+ days late), which is the standard meaning of "
        "default in consumer lending. Only 1 loan was in the 'Default' status at the snapshot, so in practice the target is "
        "'charged off vs fully paid'.",
    ]),
    *step(4, "Define the eligible population", [
        "Only loans with a final outcome are kept: <b>1,348,099 resolved loans</b> (59.6% of all rows); 912,602 unresolved loans "
        "are dropped. The class balance is 1,078,739 repaid (80.0%) vs 269,360 defaults (20.0%).",
        "A 20% default rate is imbalanced but not extreme. It means a model that always says 'repaid' is already 80% accurate, "
        "which is why accuracy is never used as the main metric in this project (see the metrics chapter).",
    ]),
    *step(5, "Column audit", [
        "For all 152 columns (151 + target) we recorded the data type, missing count and percentage, number of unique values and "
        "sample values. 40 columns are at least 80% missing and 7 are constant or single-valued. The audit table is saved in "
        "<i>reports/preprocessing_v1_audits/accepted_column_audit.csv</i>.",
        "The audit is the map for every later decision: which columns can be numbers, which are categories, which are dates "
        "stored as text, and which are nearly empty. Many 'nearly empty' columns turned out to be fields that LendingClub only "
        "started collecting in later years, which became important for the temporal split.",
    ]),
    *step(6, "Application-time vs future-time audit (the leakage review)", [
        "Every column was put in exactly one class by asking: <i>would we know this value when the application arrives?</i> Result: "
        "<b>94 application-time features</b> (income, DTI, FICO, credit-bureau history, loan amount, purpose, state...), "
        "<b>42 post-prediction columns</b> only known after the loan starts (payments received, recoveries, outstanding principal, "
        "last payment date, last FICO, last credit pull, hardship plans, debt settlement, funded amounts, issue date), "
        "4 LendingClub-derived columns kept for review (int_rate, installment, grade, sub_grade), 4 text / high-cardinality fields, "
        "1 historical date, 2 target, 2 identifier and 3 administrative columns.",
        "This is the most important step of the whole project. For example <i>recoveries</i> is non-zero almost only for "
        "defaulted loans, and <i>last_fico_range_high</i> drops sharply after a borrower stops paying; either would give a fake "
        "near-perfect model. The four LendingClub-derived columns are legal at decision time but are LendingClub's own risk "
        "opinion, so later every model is trained with them (version A, benchmark) and without them (version B, deployable).",
    ]),
    *step(7, "Remove target, leakage, identifiers and administrative columns", [
        "49 columns were dropped: 2 target, 2 identifiers (id, member_id), 3 administrative (url, policy_code, initial_list_status) "
        "and the 42 post-prediction columns. The result is <b>1,348,099 rows × 103 candidate columns</b>, and checks confirmed "
        "that no target, id or future column remained.",
        "Identifiers are dropped because a loan number carries no information about the borrower and can only add noise or "
        "leak ordering. Administrative fields describe how LendingClub listed the loan, not the borrower's risk.",
    ]),
    *step(8, "Data-quality investigation", [
        "We checked missingness, all-missing columns, constant columns, duplicate column names, columns with identical content, "
        "duplicate rows, infinite values, numeric ranges and category counts. Nothing needed to be removed at this stage, so the "
        "dataset stayed 1,348,099 × 103.",
        "Doing the checks and finding nothing is still valuable: it is documented evidence (saved audit CSVs) that the data is "
        "sound before modelling, which is exactly what a reviewer or interviewer will ask about.",
    ]),
    *step(9, "Feature semantics and representation plan", [
        "Each column was labelled numeric, categorical, date, free text or LendingClub-derived, with a planned treatment. Key "
        "decisions: <i>desc</i> (free-text loan description) is left out of the first version because it needs a separate NLP "
        "pipeline; <i>earliest_cr_line</i> (a date like 'Aug-2003') becomes a number of months; <i>emp_title</i>, <i>title</i> and "
        "<i>zip_code</i> need an encoder for high-cardinality categories.",
        "Planning the representation before coding avoids accidental mistakes such as one-hot encoding a date (which V1 later "
        "did by accident, creating 733 columns from one date).",
    ]),
    *step(10, "Missing-value strategy", [
        "Columns were grouped by how much they are missing, with a strategy for each. The rule carried into V2: every value used "
        "to fill gaps (medians, category labels) must be learned from <b>training data only</b>, never from validation or test.",
        "Learning medians from all the data would let information from the future test period leak into training. It is a "
        "small leak, but in credit risk the test must simulate the future honestly.",
    ]),
    p("Why V1 Steps 11-16 were replaced by Preprocessing V2", "h3"),
    *ps(
        "V1 originally finished with a random 80/10/10 split and 1,943 columns. A review found five problems: (1) the labels were "
        "never exported, so no model could load them; (2) earliest_cr_line was one-hot encoded into 733 columns instead of one "
        "number; (3) zip_code was one-hot encoded into 936 columns; (4) the fitted encoders and medians were never saved, so a new "
        "application could not be scored; (5) the random split mixes years, while credit models must be tested on later loans.",
        "Instead of patching, V2 rebuilt Steps 11-16 as tested Python modules and kept all V1 outputs untouched as a backup. "
        "Being able to explain why you threw work away and redid it properly is a good interview story.",
    ),
    NewPage(),
    p("Part B: Preprocessing V2 (src/preprocessing_v2/, run once with run_preprocessing_v2.py)", "h2"),
    *step(11, "Restore id, issue date and target safely", [
        "The V1 output had no id, issue date or target (Step 7 removed them on purpose). V2 re-reads the raw CSV in its original "
        "order with the same target mapping (1,348,099 rows) and re-attaches them by position, but only after a safety check: 12 "
        "shared columns (loan_amnt, int_rate, annual_inc, dti, revol_util, fico_range_low, term, grade, emp_title, title, zip_code, "
        "earliest_cr_line) are compared row by row.",
        "Result: 0 mismatches, ids unique, every issue date parses. If even one row had differed, the run would have stopped. "
        "Attaching labels by position without such a check is a classic silent bug that misaligns X and y.",
    ]),
    *step(12, "Temporal split by issue month", [
        "Train = loans issued 2012-08 to 2014-12 (388,124 loans, 17.24% default); validation = 2015 H1 (162,745, 20.35%); "
        "test = 2015 H2 (212,801, 20.06%); stress = 2016-2018 (518,744, 22.42%, optional). Loans before Aug-2012 (65,685) are "
        "excluded.",
        "Why start in Aug-2012: 33 credit-bureau fields (tot_cur_bal, num_*, mo_sin_*...) only start being filled in Mar-Aug 2012; "
        "earlier loans would teach the model that 'missing' means 'old loan'. Why not test on 2016+: at the 2018 snapshot only "
        "67.5% of 2016 loans (38% of 2017, 11% of 2018) had finished, and the finished ones are mostly early defaults and early "
        "payoffs, so their default rate is distorted. Every split still has at least 33k defaults, so no resampling is needed.",
    ]),
    *step(13, "Feature engineering (row by row, nothing learned from data)", [
        "<b>credit_history_months</b> = (application year − first-credit-line year) × 12 + (application month − first-credit-line "
        "month). It replaces the 733 date columns with one number (train range 36-842 months, median 177). "
        "<b>term_months</b>: ' 36 months' → 36. <b>emp_length_years</b>: '< 1 year' → 0 ... '10+ years' → 10. Text categories are "
        "stripped and zip_code / emp_title / title lowercased so 'Teacher' and 'teacher ' are the same.",
        "These transformations are stateless (nothing is learned), and the exact function lives inside the saved pipeline, so a "
        "new application goes through identical steps. Using the application month (not today's date) for credit history is "
        "what keeps the feature honest for historical loans.",
    ]),
    *step(14, "Fitted pipeline, trained on TRAIN only", [
        "One scikit-learn Pipeline: (a) a <b>train-only column filter</b> drops 31 columns that are >95% missing or constant in "
        "train (joint-application fields, second-applicant fields, and 14 bureau fields LendingClub only began reporting in late "
        "2015); (b) a <b>ColumnTransformer</b> with three branches: 62 numeric columns get the train median plus a 0/1 'was "
        "missing' flag; 6 categorical columns are one-hot encoded with categories under 500 training rows grouped as "
        "'infrequent'; 3 high-cardinality columns (zip_code, emp_title, title) are <b>target encoded</b>; (c) 17 duplicate "
        "'was missing' flags are dropped. Result: 103 raw columns → <b>190 numeric features</b>.",
        "<b>Target encoding</b> replaces each category with its smoothed training default rate (e.g. job title 'nurse' → 15.2%). "
        "To avoid leakage, training rows get 5-fold <b>out-of-fold</b> values (a row never sees its own label); validation and test "
        "use the train-only mapping; unseen categories get the overall train default rate (17.24%). The <b>missing flags</b> matter "
        "because 'missing' often means 'never happened' (e.g. months since last delinquency is missing for people who were never "
        "delinquent), which is useful information.",
    ]),
    *step(15, "Export", [
        "Each split is transformed (validation / test / stress in 100k-row chunks to save memory), written to Parquet, then read "
        "back and checked. X_&lt;split&gt;.parquet holds the 190 features; y_&lt;split&gt;.parquet holds id, issue_d and target in exactly "
        "the same row order. The whole fitted pipeline is saved as one file, <i>preprocessor_v2.joblib</i>, and "
        "<i>metadata_v2.json</i> records split windows, feature names, feature groups and library versions.",
        "Saving the fitted pipeline is what makes deployment possible: the web app loads the same object and transforms a raw "
        "application identically. Parquet keeps types and is ~10× smaller and faster than CSV.",
    ]),
    *step(16, "72 audits and a fresh-session test", [
        "72 automated checks in 7 groups, all passed: target alignment (20), temporal ordering (8), leakage (9), data hygiene "
        "(17: 0 NaN, 0 infinite, 0 non-numeric, 0 duplicate columns in every split), consistency (9), persistence and backups (4), "
        "engineering / readiness (5). The highest single-feature AUC is 0.674 (int_rate), below the 0.80 review flag that would "
        "suggest leakage.",
        "The fresh-session test runs in a separate Python process: it loads the saved pipeline, takes the raw CSV rows of 5 test "
        "loans and reproduces the exported features exactly (max difference 0.0), and transforms a hand-written new application "
        "with an unseen zip code and missing fields without any NaN. This proves the pipeline works outside the notebook.",
    ]),
    key("leakage audit + temporal split + fit on train only + saved pipeline + audits. If asked 'how do you know there is no "
        "leakage?', mention the 42 removed columns, out-of-fold target encoding, the single-feature AUC check (max 0.674) and the "
        "72 audits."),
    NewPage(),
]

# =====================================================================================  MODELS
lb = {}
import csv  # noqa: E402
with open(R / "leaderboard.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        lb[(r["model"], r["version"], r["split"])] = r


def v(m, ver, col="roc_auc", split="validation"):
    return float(lb[(m, ver, split)][col])


story += [
    p("6. The models, step by step (M0 → M4, then the final model)", "h1"),
    *ps(
        "The models were added in order of complexity, and each one had to beat the previous one on the validation set "
        "(2015 H1). Every model was trained in two versions: <b>A</b> uses LendingClub's grade, sub-grade, interest rate and "
        "instalment; <b>B</b> does not. LendingClub set those values from its own risk model after reviewing the application, so a "
        "real lender building its own engine would not have them. <b>B is the deployable version</b>; A is a benchmark that shows "
        "how much of LendingClub's opinion our features can rebuild. All numbers below are validation unless marked test.",
    ),
    table([
        ["Model", "Version A ROC-AUC", "Version B ROC-AUC", "B Brier", "Verdict"],
        ["M0 Dummy", "0.500", "0.500", f"{v('M0', 'B', 'brier'):.4f}", "floor"],
        ["M1 Logistic Regression", f"{v('M1', 'A'):.4f}", f"{v('M1', 'B'):.4f}", f"{v('M1', 'B', 'brier'):.4f}", "strong, simple baseline"],
        ["M2 Random Forest", f"{v('M2', 'A'):.4f}", f"{v('M2', 'B'):.4f}", f"{v('M2', 'B', 'brier'):.4f}", "no real gain (+0.002)"],
        ["M3 LightGBM", f"{v('M3', 'A'):.4f}", f"{v('M3', 'B'):.4f}", f"{v('M3', 'B', 'brier'):.4f}", "clear gain (+0.011) → chosen"],
        ["M4 XGBoost", f"{v('M4', 'A'):.4f}", f"{v('M4', 'B'):.4f}", f"{v('M4', 'B', 'brier'):.4f}", "tie, confirms M3"],
    ], [42 * mm, 30 * mm, 30 * mm, 22 * mm, W - 124 * mm]),
    Spacer(1, 4),
    p("M0: Dummy baseline", "h2"),
    *ps(
        "<b>What it is:</b> a model that ignores every feature and predicts the same PD for everyone: the training default rate, "
        "17.24% (scikit-learn DummyClassifier, strategy 'prior'). <b>Why build it:</b> it is the floor. Any real model must beat "
        "it, and its scores show what 'zero intelligence' looks like for each metric.",
        "<b>Results:</b> ROC-AUC exactly 0.500 (it cannot rank anyone), KS 0, Brier 0.1631 on validation. Its calibration gap is "
        "−3.1 points because the default rate rose from 17.2% in train to 20.4% in validation: the first sign of <b>drift over "
        "time</b>, which is why calibration had to be redone on newer data later.",
    ),
    Table([[img(R / "M0_dummy_baseline/results/m0_calibration.png", W / 2 - 4, max_h=56 * mm), img(R / "M0_dummy_baseline/results/m0_approval_curve.png", W / 2 - 4, max_h=56 * mm)]],
          colWidths=[W / 2] * 2),
    NewPage(),
    p("M1: Logistic Regression", "h2"),
    *ps(
        "<b>How it works:</b> a linear score z = b<sub>0</sub> + b<sub>1</sub>x<sub>1</sub> + ... + b<sub>n</sub>x<sub>n</sub> passed "
        "through the sigmoid PD = 1 / (1 + e<super>−z</super>). Each coefficient is the change in log-odds of default for one unit "
        "of a feature. It is the classic credit-scorecard model: fast, stable and easy to explain.",
        "<b>Preparation:</b> 26 skewed columns (income, balances, limits) were log-transformed with log1p, then everything was "
        "standardised (StandardScaler) so the coefficients are comparable and the solver converges. fico_range_high was dropped "
        "because it is always fico_range_low + 4 (perfectly collinear). <b>Tuning:</b> L2 regularisation strength C from "
        "{0.0001...1}, with or without class_weight='balanced', chosen by validation ROC-AUC. Chosen: log+scale, C = 0.001 (B), "
        "no class weights.",
        f"<b>Results:</b> validation ROC-AUC {v('M1', 'B'):.4f} (B) / {v('M1', 'A'):.4f} (A), Brier {v('M1', 'B', 'brier'):.4f}. "
        "A big jump from 0.5, which already shows the features carry real signal. Interesting fact for interviews: at a 0.5 "
        "threshold it is 80% accurate but catches only about 3% of defaulters, which is exactly why accuracy is a bad metric here.",
    ),
    Table([[img(R / "M1_logistic_regression/results/m1_coefficients.png", W / 2 - 4, max_h=56 * mm), img(R / "M1_logistic_regression/results/m1_calibration.png", W / 2 - 4, max_h=56 * mm)]],
          colWidths=[W / 2] * 2),
    p("M2: Random Forest", "h2"),
    *ps(
        "<b>How it works:</b> many decision trees, each trained on a random half of the rows and a random subset of features; the "
        "forest's PD is the average of the trees' votes. It captures non-linear effects and interactions that logistic "
        "regression cannot. <b>Tuning:</b> min_samples_leaf {20, 50, 100} and max_features {sqrt, 0.3}, 500 trees. Run on a "
        "Kaggle CPU kernel because the laptop ran out of memory.",
        f"<b>Results:</b> validation ROC-AUC {v('M2', 'B'):.4f} (B), only +0.002 over logistic regression, and a worse Brier "
        f"({v('M2', 'B', 'brier'):.4f}). Its train AUC (0.855) is far above validation, a sign of overfitting: deep trees memorise. "
        "Conclusion: non-linearity alone did not help much; boosting was needed.",
    ),
    Table([[img(R / "M2_random_forest/results/m2_importance.png", W / 2 - 4, max_h=56 * mm), img(R / "M2_random_forest/results/m2_tree_curve.png", W / 2 - 4, max_h=56 * mm)]],
          colWidths=[W / 2] * 2),
    NewPage(),
    p("M3: LightGBM (gradient boosting) — the chosen model", "h2"),
    *ps(
        "<b>How gradient boosting works:</b> trees are built one after another; each new small tree is fitted to the errors "
        "(gradients of the log loss) of all previous trees, and its prediction is added with a small weight (the learning rate). "
        "After thousands of rounds the sum of trees gives a very accurate log-odds. LightGBM grows trees leaf-wise using "
        "histograms of feature values, which makes it fast on large data.",
        "<b>Tuning:</b> random search of 40 settings per version (num_leaves 15-255, min_data_in_leaf 20-2,000, feature_fraction, "
        "bagging_fraction, L1/L2 regularisation, min_gain_to_split), learning rate 0.05 with <b>early stopping</b> (stop when "
        "validation AUC has not improved for 100 rounds), then a final fit at learning rate 0.02. Chosen for B: 21 leaves, "
        "min_data_in_leaf 1,433, feature_fraction 0.64, bagging 0.65, 2,279 rounds. Small, heavily regularised trees won.",
        f"<b>Results:</b> validation ROC-AUC <b>{v('M3', 'B'):.4f}</b> (B) / {v('M3', 'A'):.4f} (A), Brier {v('M3', 'B', 'brier'):.4f}: "
        "a clear +0.011 over logistic regression. Version B is only 0.005 behind A, so our features rebuild most of LendingClub's "
        "own risk opinion. <b>B-mono</b> adds monotonic constraints on 6 features and costs only 0.0008 AUC (0.7354).",
    ),
    Table([[img(R / "M3_lightgbm/results/m3_learning_curve.png", W / 2 - 4, max_h=56 * mm), img(R / "M3_lightgbm/results/m3_shap.png", W / 2 - 4, max_h=56 * mm)]],
          colWidths=[W / 2] * 2),
    p("M4: XGBoost — the robustness check", "h2"),
    *ps(
        "<b>Why another booster:</b> if a second, independent library gives the same result, the LightGBM result is not a fluke "
        "of one implementation. XGBoost grows trees level-wise (depth-limited) with the 'hist' method. Same search design: "
        "max_depth 3-10, min_child_weight, subsample, colsample_bytree, L1/L2, gamma; chosen depth 3 with ~2,600 rounds.",
        f"<b>Results:</b> validation ROC-AUC {v('M4', 'B'):.4f} (B) vs {v('M3', 'B'):.4f} for LightGBM: a tie (difference −0.0006, "
        "and the two models rank loans almost identically, Spearman 0.990). By the rule fixed in advance, ties go to the faster "
        "model: LightGBM trains about 2.3× faster (41 s vs 97 s per search fit). CatBoost was considered and skipped because "
        "two boosters already agreed.",
    ),
    Table([[img(R / "M4_xgboost/results/m4_learning_curve.png", W / 2 - 4, max_h=56 * mm), img(R / "M4_xgboost/results/m4_calibration.png", W / 2 - 4, max_h=56 * mm)]],
          colWidths=[W / 2] * 2),
    NewPage(),
    p("Finalising the model (models/final/, Stages 1-5)", "h2"),
    p("Stage 1: monotonic constraints and calibration", "h3"),
    *ps(
        "<b>Monotonic constraints</b> force the model to respect credit logic: PD can only go <b>down</b> as FICO or income go up, "
        "and only <b>up</b> as DTI, term, inquiries or revolving utilisation go up. We tested 1,000 validation applicants by "
        "sweeping each feature while holding everything else fixed: the unconstrained model had <b>3,709 violations</b> (e.g. "
        "99.9% of applicants had at least one income value where more income raised the PD), the constrained model <b>0</b>. "
        "Rule fixed beforehand: choose the constrained model if it costs less than 0.005 AUC; it cost 0.0008.",
        "<b>Calibration:</b> the raw model said 18.7% on average while the validation default rate was 20.35% (drift from the "
        "2012-14 training period). We compared no calibration, <b>Platt scaling</b> (a logistic regression on the logit of the PD: "
        "PD<sub>cal</sub> = 1 / (1 + e<super>−(a·logit(PD)+b)</super>), 2 numbers) and <b>isotonic regression</b> (a step-shaped "
        "monotone mapping), each scored out-of-fold with 5-fold cross-fitting inside validation. Platt won the tie-break "
        "(Brier 0.14236, simpler, saved as plain JSON: a = 1.0412, b = 0.1723) and brought the mean PD to exactly 20.35%.",
    ),
    img(ROOT / "models/final/01_selection_calibration/results/f1_calibration.png", W, max_h=70 * mm),
    p("Stage 2: fairness and reason codes", "h3"),
    *ps(
        "The data has no race, sex or age, so the main fairness risk is <b>geography as a proxy</b>: an area's default history "
        "can stand for the people who live there. Rule fixed beforehand: drop zip_code if it adds less than 0.002 AUC. It added "
        "only <b>+0.0004</b> (0.7354 with vs 0.7351 without), so it was dropped; without it, approvals shift toward the "
        "highest-default areas by +2.2 points, meaning zip had partly judged people by where they live. The final model, "
        "<b>M3-B-mono-nozip</b>, has 146 features, 0 monotonicity violations and was recalibrated (a = 1.0386, b = 0.1685).",
        "State-level check: for 33 states with ≥ 1,000 loans the calibration gap stayed within ±2.8 points; LA, TN and OR were "
        "flagged for monitoring. Reason codes come from SHAP (next chapters); the most common decline reasons were loan term "
        "(60%), FICO (56%) and DTI (43%).",
    ),
    NewPage(),
    p("Stage 3: the decision layer", "h3"),
    *ps(
        "<b>Loss model</b> from the 66,928 defaulted training loans: EAD share (principal still owed at default ÷ funded) = 57.3% "
        "for 36-month and 70.5% for 60-month loans; LGD (share of that never recovered, after collection fees) = 88.7% / 88.8%. "
        "<b>Expected loss = PD × loan amount × EAD share × LGD</b>, e.g. a $10,000 36-month loan with PD 20% → 0.2 × 10,000 × 0.573 "
        "× 0.887 ≈ $1,015.",
        "<b>Risk bands</b> R1-R7 at fixed PD cut-offs 5/10/15/20/30/40%; the actual default rate rises in every band (3.2% → 50.1%). "
        "<b>Thresholds</b> from two business targets written before looking: approve the largest group whose default rate is "
        "≤ 12%, decline the riskiest group whose default rate is ≥ 40%. Result: <b>APPROVE if PD ≤ 22.5%</b> (64.7% of validation "
        "applicants, 11.9% default), <b>DECLINE if PD ≥ 27.5%</b> (25.5%, 40.1% default), REVIEW in between (9.8%). Expected loss "
        "back-tested at 0.94× the real loss.",
    ),
    img(ROOT / "models/final/03_decision_layer/results/f3_risk_bands.png", W * 0.7, max_h=46 * mm),
    img(ROOT / "models/final/03_decision_layer/results/f3_thresholds.png", W, max_h=44 * mm),
    p("Stages 4 and 5: the scoring package and the one-time test", "h3"),
    *ps(
        "<b>Stage 4</b> packaged the preprocessor, model, calibrator and policy into a checksummed bundle with "
        "<i>CreditRiskEngine</i>. Proof of correctness: the original raw CSV rows of 2,000 validation loans reproduce Stage 3's PD, "
        "band, decision and expected loss with a difference of exactly 0.0.",
        "<b>Stage 5</b> opened the sealed 2015 H2 test set once. The pass criteria were committed to git before opening it. "
        "Results: ROC-AUC <b>0.7465</b> (validation 0.7351), KS 0.360, Brier 0.1388, mean PD 19.6% vs 20.1% actual; LendingClub's "
        "interest rate ranks the same loans at 0.7148. 7 of 8 criteria met; the miss is the 60-month calibration gap (−2.5 points, "
        "documented, not fixed, because changing the model after seeing the test would make the test meaningless). On 2016-2018 "
        "stress data the AUC stays 0.70-0.72.",
    ),
    img(ROOT / "models/final/05_final_test/results/f5_test_calibration.png", W, max_h=46 * mm),
    NewPage(),
]

# =====================================================================================  DEPLOYMENT
story += [
    p("7. Deployment", "h1"),
    img(FIG / "deployment.png", W, max_h=120 * mm),
    *ps(
        "<b>FastAPI</b> (deploy/app/main.py) is a Python web framework. It exposes <i>GET /</i> (the web page), <i>/api/health</i>, "
        "<i>/api/model</i> (model card), <i>/api/samples</i> (six real 2015 applicants) and <i>POST /api/score</i>, and generates "
        "interactive documentation at <i>/docs</i> automatically from the Pydantic request model. The engine is loaded once at "
        "start-up (FastAPI lifespan), so each request only scores, it does not reload files.",
        "<b>Docker</b> packages the app with its exact environment: the Dockerfile starts from python:3.14-slim, installs libgomp1 "
        "(OpenMP, needed by LightGBM) and the pinned library versions (scikit-learn 1.9.0 must match the version that fitted the "
        "preprocessor), and copies only the engine code, the 10 MB bundle and the app. No training data goes into the image. "
        "<b>Render</b> builds that image from GitHub using render.yaml (a 'Blueprint') and runs it as a free web service; every "
        "push to main redeploys automatically. The app uses ~240 MB of RAM, inside the free 512 MB limit.",
    ),
    p("Problems solved during deployment", "h3"),
    *bullets([
        "<b>Hugging Face Spaces</b> was the first plan, but Docker Spaces now need a paid plan (HTTP 402), so we moved to Render.",
        "<b>Line endings:</b> Git on Windows can rewrite the model text file and break its checksum, so the bundle is marked "
        "binary in .gitattributes.",
        "<b>Speed:</b> scoring took 2.5-3.4 s on Render because scikit-learn's target encoder rebuilds a lookup of 143,539 job "
        "titles on every call. Caching it as a dictionary at start-up gave identical outputs (tested to 0.0 difference) and "
        "~0.7 s per request.",
        "<b>Cold starts:</b> the free service sleeps after 15 idle minutes; the page shows a 'waking up' message.",
    ]),
    NewPage(),
    p("Guardrails added after two independent audits (v1.1 → v1.3)", "h2"),
    *ps(
        "A friend's AI agent audited the live app as a black box (~540 requests in total). It found that the model itself "
        "behaved sensibly on the main features, but that the product around it approved absurd inputs. Because the test set was "
        "already used, we did <b>not retrain</b>; instead we added a policy layer around the frozen model and measured its effect "
        "on the test set.",
    ),
    table([
        ["Audit finding", "Fix (version)", "Why"],
        ["FICO 300 was approved", "REFER if FICO < 660, DTI > 40, loan outside $1k-$35k, income < $3k... (v1.1)",
         "LendingClub never lent below 660, so the model never saw such people; its PD there means nothing."],
        ["Empty application approved", "10 core fields required (v1.1)", "A PD built on imputed values only is not a decision."],
        ["Blank loan title gave a bonus", "Title filled from purpose, like LendingClub's form (v1.1)", "A rare 'no title' group had a low default rate."],
        ["Bureau data missing → approved", "APPROVE becomes REVIEW if < 50% of bureau fields given (v1.1)", "Imputed medians look 'average-safe'."],
        ["$35k loan on $3k income approved", "REFER if loan > 50% of annual income (v1.2)", "Training max loan-to-income was exactly 0.50."],
        ["REFER still showed a PD", "Score hidden on REFER (v1.2)", "Out of range, the number is not valid."],
        ["Server errors (500) on bad input", "Strict number parsing, whole-number counts (v1.2)", "Clean 422 messages instead."],
        ["Made-up job title lowered PD", "Unseen titles treated as 'not provided' (v1.3)", "Unknowns got the overall average (17%)."],
    ], [45 * mm, 62 * mm, W - 107 * mm], RED),
    Spacer(1, 4),
    *ps("Effect on the sealed test set: the v1.2 rules change 123 of 212,801 decisions (0.06%). The v1.3 title rule touches 19% of "
        "loans (real but unseen job titles) and 5,833 decisions; ROC-AUC 0.7465 → 0.7456, mean PD 20.3% vs 20.1% actual. "
        "Not fixed (would need retraining): delinquency and credit-age effects learned from approved-only data, and state as an input."),
    NewPage(),
]

# =====================================================================================  SHAP
story += [
    p("8. SHAP: explaining each prediction", "h1"),
    *ps(
        "<b>The question SHAP answers:</b> for one applicant, how much did each feature push the prediction up or down compared "
        "with an average applicant? A model that says 'PD = 38%' is not enough in lending; the applicant has the right to know "
        "the main reasons, and the lender needs to check that the reasons make sense.",
        "<b>The idea comes from game theory (Shapley values).</b> Think of the features as players in a team and the prediction as "
        "the team's score. A feature's Shapley value is its average marginal contribution over all possible orders in which the "
        "features could join the team. This is the only way of splitting the credit that is fair in a precise mathematical sense "
        "(efficiency, symmetry, dummy and additivity properties).",
    ),
    formula("prediction(x) = base value + φ<sub>1</sub> + φ<sub>2</sub> + ... + φ<sub>n</sub>"),
    *ps(
        "Here the base value is the model's average output, and φ<sub>i</sub> is the SHAP value of feature i for this applicant. "
        "The most important property is <b>additivity</b>: the SHAP values add up exactly to the prediction. We checked this on all "
        "162,745 validation loans: the largest difference was 3.3 × 10<super>−14</super>, i.e. exact. For LightGBM the values are in "
        "<b>log-odds</b> (the model's raw output before the sigmoid), so they add up to the log-odds of default; positive values "
        "push risk up, negative values push it down.",
        "<b>How we computed it:</b> exact Shapley values are expensive in general, but for tree models there is a fast exact "
        "algorithm (TreeSHAP). LightGBM has it built in: <i>booster.predict(X, pred_contrib=True)</i> returns one value per feature "
        "plus the base value. It takes about 11 ms for one applicant, so it runs live on every request.",
    ),
    img(FIG / "shap_waterfall.png", W, max_h=95 * mm),
    p("From SHAP values to reason codes", "h3"),
    *bullets([
        "<b>Group into concepts:</b> the model has 146 columns, but several describe one idea (all addr_state_* dummies are 'State', "
        "a missing flag belongs to its feature). Because SHAP is additive, a concept's value is the sum of its columns' values (65 concepts).",
        "<b>Rank:</b> the 4 largest positive concepts are the risk reasons (the usual number in US adverse-action notices); the 2 "
        "largest negative ones are strengths.",
        "<b>Describe with the applicant's own value:</b> 'Credit score (FICO): 660', 'Debt-to-income ratio: 37.0%', 'Loan term: "
        "60 months' (src/modeling/reason_codes.py).",
        "<b>Guard (v1.2):</b> a derogatory record or a missing value is never shown as a strength, because the model learned some "
        "of those effects backwards from approved-only data.",
    ]),
    *ps("<b>Global view:</b> averaging |SHAP| over all applicants gives feature importance. The top concepts were loan term, FICO, "
        "DTI, job title, accounts opened in the last 24 months, loan amount and income."),
    img(ROOT / "models/final/02_reason_codes_fairness/results/f2_concept_importance_pdf.png", W * 0.8, max_h=85 * mm),
    NewPage(),
]

# =====================================================================================  METRICS
story += [
    p("9. Evaluation metrics: formulas and why each was used", "h1"),
    *ps(
        "Every model was scored with the same function (src/modeling/evaluation.py). Notation: y = 1 for default, 0 for repaid; "
        "p = predicted probability of default; N = number of loans. For a threshold t, a loan is 'predicted default' if p ≥ t, "
        "which gives the confusion matrix: TP (defaulter flagged), FP (good borrower flagged), TN, FN.",
    ),
    p("Why not accuracy?", "h2"),
    formula("Accuracy = (TP + TN) / N"),
    *ps("With 20% defaults, predicting 'repaid' for everyone is 80% accurate and useless. Logistic regression at threshold 0.5 "
        "was 80% accurate but caught only 3% of defaulters. Accuracy depends on one arbitrary threshold and hides the minority "
        "class, so it is reported only as a warning, never used to choose models."),
    p("ROC-AUC (main metric)", "h2"),
    formula("TPR = TP / (TP + FN)  ·  FPR = FP / (FP + TN)  ·  AUC = area under the curve of TPR vs FPR over all thresholds"),
    formula("Equivalent: AUC = P( p<sub>random defaulter</sub> &gt; p<sub>random good borrower</sub> )"),
    *ps("<b>Meaning:</b> the probability that the model gives a randomly chosen defaulter a higher PD than a randomly chosen "
        "repaid borrower. 0.5 = random, 1.0 = perfect. <b>Why:</b> lending is a ranking problem (approve the safest first); AUC "
        "measures ranking quality independent of any threshold and of the default rate, so it is comparable across periods. "
        "Typical consumer-credit models on application data score 0.70-0.80, so 0.746 is realistic, not suspicious."),
    p("PR-AUC (average precision)", "h2"),
    formula("Precision = TP / (TP + FP)  ·  Recall = TPR  ·  AP = Σ<sub>k</sub> (R<sub>k</sub> − R<sub>k−1</sub>) · P<sub>k</sub>"),
    *ps("<b>Why:</b> it focuses on the minority class (defaulters). Its baseline is the default rate (0.20), not 0.5, so our "
        "0.43 on test is about 2.2× better than random. Useful as a second view of ranking on imbalanced data."),
    p("KS statistic (Kolmogorov-Smirnov)", "h2"),
    formula("KS = max<sub>t</sub> | F<sub>default</sub>(t) − F<sub>repaid</sub>(t) | = max<sub>t</sub> (TPR(t) − FPR(t))"),
    *ps("<b>Meaning:</b> the largest gap between the score distributions of defaulters and repaid borrowers. <b>Why:</b> it is the "
        "traditional metric in banking credit scorecards, so lenders and interviewers from banks recognise it. 0.36 on test "
        "is good for application scoring."),
    p("Brier score", "h2"),
    formula("Brier = (1/N) Σ<sub>i</sub> (p<sub>i</sub> − y<sub>i</sub>)<super>2</super>"),
    *ps("<b>Meaning:</b> mean squared error of the probabilities; lower is better. It rewards both ranking and correct "
        "probability levels. <b>Why:</b> our PD feeds expected loss in dollars, so the probability itself must be right, not just "
        "the order. Used to choose the calibration method."),
    p("Log loss (binary cross-entropy)", "h2"),
    formula("LogLoss = −(1/N) Σ<sub>i</sub> [ y<sub>i</sub> ln p<sub>i</sub> + (1 − y<sub>i</sub>) ln(1 − p<sub>i</sub>) ]"),
    *ps("<b>Why:</b> it is the loss the boosting models actually minimise, and it punishes confident wrong predictions heavily. "
        "Watching it on validation shows overfitting."),
    p("Calibration gap and ECE", "h2"),
    formula("Calibration gap = mean(p) − mean(y)  ·  ECE = Σ<sub>b</sub> (n<sub>b</sub>/N) · | mean(p)<sub>b</sub> − mean(y)<sub>b</sub> |"),
    *ps("<b>Meaning:</b> the gap checks the overall level (we got −0.4 points on test); ECE (expected calibration error) splits "
        "loans into 10 equal groups by PD and averages the absolute gap per group (0.6% on test). <b>Why:</b> 'PD = 20%' must "
        "mean 20 in 100 default, otherwise the decision thresholds and expected loss are wrong."),
    p("Approval rate vs bad rate (business metric)", "h2"),
    formula("for approval rate a: approve the a% lowest PDs  →  bad rate(a) = defaults among approved / approved"),
    *ps("<b>Why:</b> this is how a lender reads a model: 'if we approve 65%, what default rate do we get?'. It was used to pick "
        "the approve threshold (approved book ≤ 12% default)."),
    p("Expected loss and its back-test", "h2"),
    formula("EL = PD × EAD × LGD  ·  EAD = amount × EAD share  ·  back-test ratio = Σ EL / Σ realised loss"),
    *ps("<b>Why:</b> it turns the probability into money. A ratio near 1.0 (0.94 on validation, 0.89 on test) means the dollar "
        "forecasts are trustworthy at portfolio level."),
    NewPage(),
]

# =====================================================================================  RESULTS CHART
story += [
    p("10. Results across all models", "h1"),
    img(FIG / "auc_ladder.png", W),
    table([
        ["Model / step", "ROC-AUC", "Note"],
        ["M0 Dummy", "0.500", "floor: same PD for everyone"],
        ["M1 Logistic Regression (B)", f"{v('M1', 'B'):.4f}", "linear scorecard, log+scale, C = 0.001"],
        ["M2 Random Forest (B)", f"{v('M2', 'B'):.4f}", "+0.002, overfits (train 0.855)"],
        ["M3 LightGBM (B)", f"{v('M3', 'B'):.4f}", "+0.011 over logistic → chosen"],
        ["M4 XGBoost (B)", f"{v('M4', 'B'):.4f}", "tie with M3 (Spearman 0.990)"],
        ["M3 B-mono", "0.7354", "monotonic constraints, 0 violations, cost 0.0008"],
        ["Final M3-B-mono-nozip", "0.7351", "zip dropped, recalibrated (out-of-fold validation)"],
        ["Final on sealed TEST", "0.7465", "212,801 loans, opened once; LendingClub rate 0.7148"],
        ["Stress 2016 / 2017 / 2018", "0.719 / 0.711 / 0.704", "ranking only (biased labels)"],
    ], [60 * mm, 38 * mm, W - 98 * mm]),
    Spacer(1, 6),
    *ps("Version A (with LendingClub's grade and rate) is always 0.005-0.008 higher on validation, which means our features "
        "rebuild most of LendingClub's own risk opinion without using it. The test score is higher than validation for every "
        "model, including LendingClub's own interest rate (0.704 → 0.715), so the 2015 H2 loans were simply easier to rank; it "
        "is not leakage."),
    NewPage(),
]

# =====================================================================================  INTERVIEW PREP
QA = [
    ("Resume: \"Built an end-to-end credit-risk engine on 1.35M LendingClub loans\"",
     [("Why 1.35M and not 2.26M?", "Only loans with a final outcome (Fully Paid or Charged Off) can be labelled. 912,602 loans were "
       "still Current or Late at the 2018 snapshot, so they were dropped. 1,348,099 remain, 20% defaults."),
      ("What was in the leakage audit of 151 columns?", "Each column was classified by 'is it known when the application arrives?'. "
       "42 post-loan columns were removed (payments, recoveries, last FICO, settlement, hardship...), plus target, ids and admin "
       "fields: 151 → 103 candidates. The highest single-feature AUC afterwards was 0.674 (interest rate), below a 0.80 leak flag."),
      ("What is an out-of-time split and why use it?", "Train on loans issued 2012-08 to 2014-12, validate on 2015 H1, test on 2015 H2. "
       "A random split lets the model see the future period during training and overstates quality; credit models are used on "
       "future applicants, so they must be tested that way."),
      ("What were the 72 audits?", "Target alignment, temporal ordering, leakage, data hygiene (no NaN/inf), consistency of features "
       "across splits, persistence of the saved pipeline, and engineering checks, plus a fresh-process test that raw rows reproduce "
       "the exported features exactly.")]),
    ("Resume: \"Benchmarked 5 models; monotonic, calibrated LightGBM; ROC-AUC 0.746 vs 0.715\"",
     [("Which 5 models and why that order?", "Dummy (floor), Logistic Regression (simple explainable baseline), Random Forest "
       "(non-linearity), LightGBM (boosting), XGBoost (a second booster to confirm). Each had to beat the previous one."),
      ("Why LightGBM over XGBoost?", "They tied (0.7362 vs 0.7356 validation, rankings 99% correlated). By a rule fixed beforehand, "
       "a tie goes to the faster model; LightGBM was 2.3× faster."),
      ("What does 'never sees the lender's grade or rate' mean?", "Grade, sub-grade, interest rate and instalment are LendingClub's own "
       "risk opinion, set after its review. A lender building its own engine would not have them, so the deployable version B "
       "excludes them. Version A with them scored only ~0.005 higher."),
      ("What are monotonic constraints?", "Rules that force PD to move one way: down with FICO and income, up with DTI, term, "
       "inquiries and utilisation. Without them the model had 3,709 logic violations on 1,000 test applicants; with them 0, at a "
       "cost of 0.0008 AUC."),
      ("What does calibrated mean and how?", "The predicted PD matches the real default rate. The raw model said 18.7% on average vs "
       "20.35% actual (drift since training). Platt scaling (2-parameter logistic on the logit) fixed it, chosen over isotonic by "
       "out-of-fold Brier score."),
      ("Is 0.746 good? Why is test higher than validation?", "Consumer credit models on application data usually score 0.70-0.80. "
       "Test was higher for every model including LendingClub's own rate (0.704 → 0.715), so 2015 H2 was easier to rank, not leakage. "
       "The test set was sealed in code and the criteria committed to git before it was opened once."),
      ("What is the 0.715?", "The ROC-AUC you get by ranking the same test loans by LendingClub's interest rate, i.e. LendingClub's own "
       "pricing. Our model beats it without using it. Caveat: both are measured only on loans LendingClub approved.")]),
    ("Resume: \"Decision layer: risk bands, thresholds, expected loss (PD × EAD × LGD), SHAP; 11.6% vs 20.1%; 13.6% → 6.2%\"",
     [("How were the thresholds chosen?", "From two business targets written before looking at results, searched on validation with "
       "out-of-fold PDs: approve the largest group with default rate ≤ 12% (PD ≤ 22.5%), decline the riskiest group with default "
       "rate ≥ 40% (PD ≥ 27.5%), review in between."),
      ("Explain PD × EAD × LGD.", "PD = chance of default. EAD = how much is still owed when it happens (57% of the loan for 36-month, "
       "71% for 60-month, from defaulted training loans). LGD = share of that never recovered (~89%). Their product is the expected "
       "loss; on validation it matched real losses at 0.94×."),
      ("What do 11.6% vs 20.1% and 13.6% → 6.2% mean?", "On the test set, the loans the engine approves (66.8% of applicants) defaulted "
       "at 11.6%, against 20.1% for all loans LendingClub funded. Realised loss per dollar lent falls from 13.6% (approve everyone) "
       "to 6.2% (approve only APPROVE). Caveat: lost interest from rejected good borrowers is not counted."),
      ("What is SHAP?", "Shapley values from game theory: each feature's fair share of the difference between this prediction and the "
       "average. They add up exactly to the model output (checked to 10^-14). We group them into 65 concepts and show the top 4 as "
       "plain-language reasons.")]),
    ("Resume: \"FastAPI + Docker; fairness ablation; audit-driven guardrails\"",
     [("Walk me through a request.", "Browser → Render → Docker container → uvicorn → FastAPI → validation and guardrails → "
       "CreditRiskEngine (preprocessor → LightGBM → Platt → policy → SHAP) → JSON. ~0.7 s on the free server."),
      ("What is the checksummed bundle?", "The four model files plus a manifest with SHA-256 hashes and library versions. The engine "
       "refuses to start if a file changed or scikit-learn is not 1.9.0, so the live app scores exactly like the tested model."),
      ("What was the fairness ablation?", "Train with and without zip code. It added only +0.0004 AUC, below the 0.002 rule, so it was "
       "dropped because it can act as a proxy for where people live. Approvals then shifted +2.2 points toward high-default areas."),
      ("What guardrails and why didn't you retrain?", "Two black-box audits found absurd approvals (FICO 300, $35k on $3k income, "
       "empty forms). Fix: REFER out-of-range applicants with no score shown, required fields, strict validation, affordability "
       "rule, no auto-approval without bureau data, unknown titles treated as missing. No retraining because the test set was "
       "already used; a retrained model could not be honestly re-tested. The v1.2 rules changed 0.06% of test decisions.")]),
    ("Likely follow-up questions",
     [("Biggest limitation?", "Selection bias: trained only on approved loans, so it never saw people LendingClub rejected (fixed "
       "partly with REFER rules). Also 60-month loans are under-predicted by 2.5 points on test, and delinquency effects are weak."),
      ("How would you improve it?", "Reject inference using the rejected-applications file; a term-specific calibrator on newer data; "
       "retrain without free-text titles and with more monotonic constraints; monitoring for drift (PSI) and periodic recalibration."),
      ("Why not deep learning?", "On tabular data with ~150 features, gradient-boosted trees are the state of the art, train fast on CPU "
       "and give exact SHAP values; neural nets rarely beat them here and are harder to constrain and explain."),
      ("How did you handle class imbalance?", "No resampling: 20% defaults is enough signal (33k+ defaults per split). class_weight was "
       "tested and rejected because it distorts probabilities, and we need calibrated PDs."),
      ("What would you monitor in production?", "Input drift (PSI per feature), the default rate of approved loans vs predicted, "
       "calibration by segment (term, state), approval rate, REFER rate and latency.")]),
]
story += [p("11. Interview preparation: questions from the resume", "h1"),
          p("Every number on the resume, with the question an interviewer is likely to ask and a short answer you can give.", "small")]
for section_title, qas in QA:
    story.append(p(section_title, "h2"))
    for q, a in qas:
        story.append(KeepTogether([p("Q: " + q, "q"), p(a)]))
story += [
    Spacer(1, 6),
    box([p("<b>Numbers to remember:</b> 2.26M loans → 1.35M resolved (20% default) · 151 → 103 → 190 → 146 features · 42 leakage "
           "columns · 72 audits · train 388,124 / val 162,745 / test 212,801 · ROC-AUC 0.736 (val) / 0.746 (test) vs 0.715 · "
           "3,709 → 0 violations · zip +0.0004 → dropped · Platt a ≈ 1.04 · approve PD ≤ 22.5%, decline ≥ 27.5% · 66.8% approved at "
           "11.6% default · loss 13.6% → 6.2% · EAD 57% / 71%, LGD ~89% · ~0.7 s per request · guardrails changed 0.06% of test decisions.")],
        PURPLE, "#f3effc"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "AI Credit Risk & Loan Approval Engine · project guide")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm, bottomMargin=16 * mm,
                  title="AI Credit Risk & Loan Approval Engine - Project Guide", author="Mitansh Kanani").build(
    story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
