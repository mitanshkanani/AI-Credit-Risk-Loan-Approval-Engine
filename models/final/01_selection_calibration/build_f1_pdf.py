"""Build models/final/01_selection_calibration/F1_selection_calibration.pdf from the saved Stage 1 results.

    python models/final/01_selection_calibration/build_f1_pdf.py
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "F1_selection_calibration.pdf"

choice = json.loads((RESULTS / "final_model_choice.json").read_text(encoding="utf-8"))
cal = json.loads((RESULTS / "calibrator.json").read_text(encoding="utf-8"))
mono = pd.read_csv(RESULTS / "f1_monotonicity.csv")
comp = pd.read_csv(RESULTS / "f1_calibration_comparison.csv").set_index("method")
method = choice["calibration_method"]

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
GREY, BLUE, GREEN, ORANGE = (colors.HexColor(c) for c in ("#4b5563", "#3b6fd4", "#2e9b5b", "#d4840f"))
S = {
    "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=19, leading=23, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Body", fontSize=10, leading=14, textColor=MUTED, spaceAfter=6),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=6, spaceAfter=3),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.3, leading=12.8, textColor=INK, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.2, leading=12.6, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=2),
    "c": ParagraphStyle("c", fontName="Body", fontSize=8.5, leading=11.2, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-Bold", fontSize=8.6, leading=11.2, textColor=colors.white),
    "small": ParagraphStyle("sm", fontName="Body", fontSize=7.9, leading=10.4, textColor=MUTED),
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
             ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6)]
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


auc = choice["validation_roc_auc"]
viol = choice["monotonicity_violations"]
labels = {"fico_range_low": "Higher FICO → lower PD", "dti": "Higher DTI → higher PD", "annual_inc": "Higher income → lower PD",
          "term_months": "60-month term → higher PD", "inq_last_6mths": "More recent inquiries → higher PD",
          "revol_util": "Higher card utilisation → higher PD"}
mono_rows = [["Rule the model must follow", "M3-B: applicants violating", "M3-B: worst wrong move", "M3-B-mono: applicants violating"]]
for feature, text in labels.items():
    b = mono[(mono.model == "M3-B") & (mono.feature == feature)].iloc[0]
    m = mono[(mono.model == "M3-B-mono") & (mono.feature == feature)].iloc[0]
    mono_rows.append([text, f"{b.applicants_with_violation} / 1,000 ({b.share:.0%})", f"{b.largest_wrong_step_pd_points:.2f} PD points",
                      f"<b>{m.applicants_with_violation}</b> / 1,000"])

fmt_rows = [["Method", "ROC-AUC", "Brier", "Log loss", "Mean PD (actual 20.35%)", "Calibration error (ECE)", "Gap 36-month", "Gap 60-month"]]
for m_name, label in (("none", "Raw LightGBM PD"), ("platt", "Platt scaling"), ("isotonic", "Isotonic regression")):
    r = comp.loc[m_name]
    name = f"<b>{label}</b>" if m_name == method else label
    fmt_rows.append([name, f"{r.roc_auc:.4f}", f"{r.brier:.5f}", f"{r.log_loss:.4f}", pct(r.mean_pd), f"{r.ece:.4f}",
                     f"{r.gap_36m * 100:+.2f} pts", f"{r.gap_60m * 100:+.2f} pts"])

story = [
    p("Final model · Stage 1: model choice and calibration", "title"),
    p("Finalizing the credit-risk engine · validation split only · test still sealed", "sub"),
    callout([p(f"<b>Result:</b> the final model is <b>{choice['chosen_model']}</b> (LightGBM with monotonic constraints), calibrated with "
               f"<b>{'Platt scaling' if method == 'platt' else method}</b>. The constraints cost only {choice['constraint_cost']:.4f} "
               f"ROC-AUC, yet they remove {viol['M3-B']:,} cases where the unconstrained model broke basic credit logic. "
               f"Calibration lifts the average PD from {pct(comp.loc['none', 'mean_pd'])} to {pct(comp.loc[method, 'mean_pd'])}, "
               f"matching the real {pct(comp.loc[method, 'actual_rate'])} default rate.")], GREEN, "#ecf8f0"),
    p("1. Integrity check", "h"),
    p(f"Both models were trained on Kaggle (LightGBM 4.6) and reloaded locally (LightGBM {choice['environment']['lightgbm']}). Their "
      "validation PDs match the predictions saved on Kaggle to within 1e-16, so the saved model files are exactly the models we evaluated."),
    p("2. Do the monotonic constraints really hold?", "h"),
    p("For 1,000 random validation applicants, each rule's feature was swept across its 1st-99th percentile range (up to 25 values) "
      "while everything else stayed fixed. An applicant counts as a violation if their PD ever moves the wrong way."),
    table(mono_rows, [56 * mm, 38 * mm, 34 * mm, W - 128 * mm], BLUE),
    Spacer(1, 3),
    *bullets([
        "The unconstrained model breaks credit logic for almost everyone somewhere: for nearly every applicant there is an income "
        "range where <b>earning more raises their PD</b> (by up to "
        f"{mono[(mono.model == 'M3-B') & (mono.feature == 'annual_inc')].largest_wrong_step_pd_points.iloc[0]:.1f} points). These wiggles are noise "
        "the trees picked up, not real risk, and they would be impossible to defend to a customer or regulator.",
        "The monotonic model has <b>zero</b> violations across all six rules, for a cost of only "
        f"{choice['constraint_cost']:.4f} ROC-AUC ({auc['B']:.4f} → {auc['B-mono']:.4f}).",
    ]),
    p("3. Model choice", "h"),
    p(f"Rule fixed in advance: <i>{choice['rule']}</i>. B-mono costs {choice['constraint_cost']:.4f} (&lt; 0.005) and has "
      f"{viol['M3-B-mono']} violations, so <b>{choice['chosen_model']}</b> is the final model ({choice['n_features']} features, "
      "no LendingClub grade features)."),
    PageBreak(),
    p("4. Calibration: making the PD an honest probability", "h"),
    p("Raw LightGBM PDs run too low on 2015 loans (the drift seen since M0). Three options were compared <b>out-of-fold</b>: "
      "5-fold cross-fitting inside validation, so no loan's calibrated PD comes from a calibrator that saw that loan."),
    table(fmt_rows, [29 * mm, 21 * mm, 16 * mm, 15 * mm, 22 * mm, 21 * mm, 19 * mm, W - 143 * mm], BLUE),
    Spacer(1, 4),
    Image(str(RESULTS / "f1_calibration.png"), width=W, height=W * 0.4),
    Spacer(1, 2),
    p(f"Rule: lowest out-of-fold Brier score; Platt wins ties within 0.00005 because it is simpler. <b>{method.capitalize()}</b> "
      f"was chosen (Brier {comp.loc['none', 'brier']:.5f} → {comp.loc[method, 'brier']:.5f})."),
    callout([p(f"<b>The calibrator (saved as plain JSON, no pickle):</b> PD<sub>cal</sub> = 1 / (1 + e<super>−(a·logit(PD<sub>raw</sub>) + b)</super>) "
               f"with <b>a = {cal['a']:.4f}</b> and <b>b = {cal['b']:.4f}</b>. With a ≈ 1, it keeps the model's spread and mainly "
               "shifts all PDs up to the 2015 default level. Because it is monotone, the ranking (ROC-AUC) does not change.")] if method == "platt"
            else [p(f"<b>The calibrator</b> ({method}) is saved as plain JSON breakpoints.")], BLUE, "#eef4ff"),
    p("What we learned", "h"),
    *bullets([
        f"<b>Calibration error falls about {comp.loc['none', 'ece'] / comp.loc[method, 'ece']:.0f}×</b> (ECE {comp.loc['none', 'ece']:.4f} → "
        f"{comp.loc[method, 'ece']:.4f}); the mean PD now matches the actual default rate.",
        f"Isotonic is even closer per decile (ECE {comp.loc['isotonic', 'ece']:.4f}) but scores a slightly worse Brier and log loss and "
        f"lowers ROC-AUC to {comp.loc['isotonic', 'roc_auc']:.4f} (its step function creates ties), so Platt is the better trade-off.",
        f"<b>Remaining gap:</b> 60-month loans are still under-predicted by {-comp.loc[method, 'gap_60m'] * 100:.1f} points and 36-month "
        f"loans over-predicted by {comp.loc[method, 'gap_36m'] * 100:.1f}. One global calibrator cannot fix both; a term-specific "
        "calibrator is an option if the test set confirms the gap.",
        "<b>Honesty note:</b> validation has now been used to stop boosting, choose settings and fit the calibrator. Only the sealed test "
        "set (Stage 5) can give an unbiased final score; that is why it stays closed.",
    ]),
    Spacer(1, 3),
    p("Next: Stage 2, turning SHAP values into plain-language reason codes, plus fairness checks across states and zip regions.", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/final/01_selection_calibration · see F1_selection_calibration.ipynb for the executed code")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=14 * mm,
                  bottomMargin=16 * mm, title="Final model - Stage 1").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
