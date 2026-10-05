"""Build models/M0_dummy_baseline/M0_dummy_baseline.pdf from the saved M0 results.

    python models/M0_dummy_baseline/build_m0_pdf.py
"""

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "M0_dummy_baseline.pdf"
summary = json.loads((RESULTS / "m0_metrics.json").read_text(encoding="utf-8"))
segments = pd.read_csv(RESULTS / "m0_segment_metrics.csv")
train, val = summary["metrics"]["train"], summary["metrics"]["validation"]

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFont(TTFont("Mono", str(FONTS / "consola.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
GREY, BLUE, ORANGE = colors.HexColor("#4b5563"), colors.HexColor("#3b6fd4"), colors.HexColor("#d4840f")
S = {
    "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Body", fontSize=10, leading=14, textColor=MUTED, spaceAfter=8),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=6, spaceAfter=4),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.4, leading=13, textColor=INK, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.4, leading=13, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=2),
    "c": ParagraphStyle("c", fontName="Body", fontSize=8.6, leading=11.4, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-Bold", fontSize=8.8, leading=11.4, textColor=colors.white),
    "small": ParagraphStyle("sm", fontName="Body", fontSize=8, leading=10.5, textColor=MUTED),
}
W = A4[0] - 34 * mm


def p(text, style="b"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["li"], bulletText="•") for t in items]


def table(rows, widths, color=GREY):
    data = [[Paragraph(str(c), S["cw" if r == 0 else "c"]) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths)
    style = [("BACKGROUND", (0, 0), (-1, 0), color), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9ca3af")),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f9fafb")) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def callout(flowables, color, fill):
    t = Table([[flowables]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def pct(x):
    return f"{x * 100:.2f}%"


seg = {(r.split, int(r.term_months)): r for r in segments.itertuples()}
env = summary["environment"]

story = [
    p("M0 · Dummy baseline", "title"),
    p("Model 1 of the ladder · the floor every real model must beat · AI Credit Risk / Loan Approval Engine", "sub"),
    callout([p("<b>In one sentence:</b> M0 ignores every feature and predicts the same probability of default for "
               f"<b>every</b> applicant: the training default rate, <b>{pct(summary['learned_prior_default'])}</b>. "
               "It cannot tell a safe borrower from a risky one, and that is exactly the point: it shows what "
               "\"zero intelligence\" scores, so we can measure how much each real model adds.")],
            BLUE, "#eef4ff"),
    p("How it works", "h"),
    *bullets([
        "<b>Training</b> = counting. It looks only at the training labels (388,124 loans, 66,928 defaults) and stores "
        "the share of defaults. No features, no fitting of weights or trees.",
        "<b>Prediction</b> = returning that stored number. predict_proba gives [P(repay), P(default)] = "
        f"[{1 - summary['learned_prior_default']:.4f}, {summary['learned_prior_default']:.4f}] for everyone.",
        "<b>Versions A and B are identical</b>: B removes LendingClub's grade / sub_grade / int_rate / installment, "
        "but M0 uses no features at all.",
    ]),
    p("Parameters", "h"),
    table([
        ["Parameter", "Value used", "What it means"],
        ["strategy", "\"prior\"", "Predict the class frequencies seen in training as probabilities. Chosen because it gives "
                                  "honest probabilities (PD = 17.24%), which the probability metrics (Brier, log loss) need."],
        ["constant", "None (unused)", "Only used by strategy=\"constant\" (always predict one fixed class)."],
        ["random_state", "None (unused)", "Only used by the random strategies (\"stratified\", \"uniform\")."],
    ], [26 * mm, 26 * mm, W - 52 * mm]),
    Spacer(1, 4),
    table([
        ["Other strategy", "Why not used"],
        ["most_frequent", "Same predicted class (repay) but its probabilities are 0 / 1, which break Brier and log loss."],
        ["stratified / uniform", "Random guesses: results change with the seed and add noise without information."],
        ["constant", "Forcing one class is the same accuracy trap as most_frequent."],
    ], [30 * mm, W - 30 * mm]),
    p("Data and rules", "h"),
    *bullets([
        "Fitted on <b>train</b> (Aug-2012 to Dec-2014); reported on <b>train</b> and <b>validation</b> (2015 H1).",
        "The <b>test set stays sealed</b> until the final model is chosen; the 2016-2018 stress set is not used.",
        f"Run in Jupyter: Python {env['python']}, pandas {env['pandas']}, scikit-learn {env['scikit_learn']}.",
    ]),
    p("Results", "h"),
    table([
        ["Split", "Rows", "Actual default rate", "Predicted PD", "ROC-AUC", "KS", "PR-AUC", "Brier", "Log loss"],
        ["Train", f"{train['rows']:,}", pct(train["default_rate"]), pct(train["mean_pd"]), f"{train['roc_auc']:.3f}",
         f"{train['ks']:.3f}", f"{train['pr_auc']:.4f}", f"{train['brier']:.4f}", f"{train['log_loss']:.4f}"],
        ["Validation", f"{val['rows']:,}", pct(val["default_rate"]), pct(val["mean_pd"]), f"{val['roc_auc']:.3f}",
         f"{val['ks']:.3f}", f"{val['pr_auc']:.4f}", f"{val['brier']:.4f}", f"{val['log_loss']:.4f}"],
    ], [18 * mm, 18 * mm, 22 * mm, 19 * mm, 19 * mm, 12 * mm, 18 * mm, 15 * mm, W - 141 * mm], BLUE),
    p("What we learned", "h"),
    *bullets([
        "<b>No ranking ability.</b> ROC-AUC 0.500 and KS 0.000. Approving the \"safest\" 10% or 90% of applicants gives the "
        f"same {pct(val['default_rate'])} bad rate, a flat approval curve.",
        f"<b>The accuracy trap.</b> M0 scores <b>{val['accuracy_at_threshold']:.1%}</b> accuracy on validation while catching "
        "<b>0%</b> of defaults. This is why accuracy is never used to judge models in this project.",
        f"<b>Drift.</b> The constant {pct(val['mean_pd'])} under-predicts the real {pct(val['default_rate'])} in 2015 H1 by "
        f"{-val['calibration_gap'] * 100:.2f} points, so the final model must be calibrated on validation.",
        f"<b>Loan term alone matters.</b> On validation, 60-month loans default at {pct(seg[('validation', 60)].default_rate)} "
        f"vs {pct(seg[('validation', 36)].default_rate)} for 36-month loans, yet M0 gives both {pct(val['mean_pd'])}. "
        "Even one feature would beat M0.",
    ]),
    Spacer(1, 4),
    Table([[Image(str(RESULTS / "m0_calibration.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.6),
            Image(str(RESULTS / "m0_approval_curve.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.6)]],
          colWidths=[W / 2, W / 2]),
    Spacer(1, 6),
    callout([p(f"<b>The bar M1 (Logistic Regression) must clear on validation:</b> ROC-AUC above 0.500, KS above 0, "
               f"PR-AUC above {val['pr_auc']:.3f}, Brier below {val['brier']:.4f} and log loss below {val['log_loss']:.4f}. "
               "Beating M0 is expected; the size of the gap is what shows how much the features are worth.")],
            ORANGE, "#fff4e5"),
    Spacer(1, 6),
    p("<b>Metric glossary.</b> ROC-AUC: chance a random defaulter gets a higher PD than a random payer (0.5 = coin flip). "
      "KS: largest gap between the score distributions of defaulters and payers. PR-AUC: precision vs recall for defaults "
      "(baseline = default rate). Brier: mean squared error of the probabilities (lower is better). Log loss: penalty for "
      "confident wrong probabilities (lower is better).", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/M0_dummy_baseline · see M0_dummy_baseline.ipynb for the executed code")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=14 * mm,
                  bottomMargin=16 * mm, title="M0 - Dummy baseline").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
