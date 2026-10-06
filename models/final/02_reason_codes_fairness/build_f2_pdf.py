"""Build models/final/02_reason_codes_fairness/F2_reason_codes_fairness.pdf from the saved Stage 2 results.

    python models/final/02_reason_codes_fairness/build_f2_pdf.py
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
OUT = HERE / "F2_reason_codes_fairness.pdf"

choice = json.loads((RESULTS / "final_model_choice.json").read_text(encoding="utf-8"))
ablation = pd.read_csv(RESULTS / "f2_zip_ablation.csv").set_index("model")
zips = pd.read_csv(RESULTS / "f2_zip_groups.csv")
states = pd.read_csv(RESULTS / "f2_states.csv")
freq = pd.read_csv(RESULTS / "f2_reason_frequency.csv")
examples = pd.read_csv(RESULTS / "f2_examples.csv")
zd = choice["zip_code_decision"]
oof = choice["validation_oof"]

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
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.3, leading=15.5, textColor=INK, spaceBefore=5, spaceAfter=3),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.1, leading=12.4, textColor=INK, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.0, leading=12.2, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=1.8),
    "c": ParagraphStyle("c", fontName="Body", fontSize=8.3, leading=10.8, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-Bold", fontSize=8.4, leading=10.8, textColor=colors.white),
    "small": ParagraphStyle("sm", fontName="Body", fontSize=7.8, leading=10.2, textColor=MUTED),
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
             ("TOPPADDING", (0, 0), (-1, -1), 2.3), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.3)]
    style += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f9fafb")) for r in range(2, len(rows), 2)]
    t.setStyle(TableStyle(style))
    return t


def callout(flowables, color, fill, width=None):
    t = Table([[flowables]], colWidths=[width or W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, color),
                           ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 5),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    return t


def pct(x):
    return f"{x * 100:.1f}%"


full = ablation.iloc[0]
flagged = states[states["flag"]]
dropped = zd["dropped"]
headline = (
    f"<b>Result:</b> <b>zip_code is dropped</b>. It added only {zd['zip_gain']:+.4f} ROC-AUC (rule: keep only if ≥ 0.002), so a "
    f"location-based input is not worth its fair-lending risk. The final model is now <b>{choice['chosen_model']}</b> "
    f"({choice['n_features']} features, still 0 monotonicity violations, recalibrated)."
    if dropped else
    f"<b>Result:</b> <b>zip_code is kept</b>: it adds {zd['zip_gain']:+.4f} ROC-AUC (≥ 0.002). The final model stays "
    f"<b>{choice['chosen_model']}</b>."
) + (f" Every applicant now gets up to 4 plain-language reasons; the most common one for declined applicants is "
     f"<b>{freq.iloc[0]['reason']}</b>.")

ab_rows = [["Model (same recipe, trained on Kaggle)", "Features", "ROC-AUC", "KS", "Brier", "Rounds"]]
for name, r in ablation.iterrows():
    ab_rows.append([name, int(r.features), f"{r.roc_auc:.4f}", f"{r.ks:.4f}", f"{r.brier:.5f}", int(r.rounds)])

zip_rows = [["Area group (by area's historical default rate)", "Loans", "Actual default rate", "Approved with zip",
             "Approved without zip", "Change"]]
for r in zips.itertuples():
    zip_rows.append([r.area_group, f"{r.loans:,}", pct(r._3), pct(r._4), pct(r._5), f"{r._6:+.1f} pts"])

high_change, low_change = zips.iloc[-1, -1], zips.iloc[0, -1]
if high_change > 0 > low_change:
    zip_reading = (f"Read: without zip_code, approvals shift toward the highest-default areas ({high_change:+.1f} pts) and away from the "
                   f"lowest ({low_change:+.1f} pts). With it, applicants are partly judged by <i>where they live</i>.")
elif high_change < 0 < low_change:
    zip_reading = (f"Read: without zip_code, the highest-default areas would be approved slightly less often ({high_change:+.1f} pts); "
                   "the other inputs already capture their higher risk.")
else:
    zip_reading = "Read: removing zip_code barely moves approval rates between area groups."

# Redraw the concept-importance chart from its CSV (cleaner title than the notebook's PNG)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
imp = pd.read_csv(RESULTS / "f2_concept_importance.csv").head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.barh(imp["label"], imp["mean_abs_shap"], color="#3b6fd4")
ax.set_xlabel("mean |SHAP| (log-odds)")
ax.set_title("Top 15 concepts (mean |SHAP|)", fontsize=11, loc="left")
plt.tight_layout()
CONCEPT_PNG = RESULTS / "f2_concept_importance_pdf.png"
plt.savefig(CONCEPT_PNG, dpi=150)
plt.close(fig)

reason_rows = [["Reason (concept)", "Share of declined applicants with it in their top 4"]]
for r in freq.head(10).itertuples():
    reason_rows.append([r.reason, pct(r.share_of_declined_applicants)])


def example_block(label):
    part = examples[examples.example == label]
    head = part.iloc[0]
    lines = [p(f"<b>{label.capitalize()}</b>: PD {pct(head.pd_calibrated)} (actually {'defaulted' if head.defaulted else 'repaid'})", "c")]
    for r in part.itertuples():
        sign = "▲" if r.type == "raises risk" else "▼"
        lines.append(p(f"{sign} {r.reason}", "c"))
    return lines


story = [
    p("Final model · Stage 2: fairness checks and reason codes", "title"),
    p("Validation split only · run on Kaggle (private CPU notebook) · test still sealed", "sub"),
    callout([p(headline)], GREEN, "#ecf8f0"),
    p("A. Does the model need zip_code?", "h"),
    p("The data has no race, sex or age, so geography is the main <b>proxy</b> to check: an area's default history can stand in for the "
      "people who live there. Three models were trained with the identical recipe (M3's parameters, learning rate 0.02, monotonic "
      "constraints, early stopping):"),
    table(ab_rows, [62 * mm, 20 * mm, 22 * mm, 20 * mm, 22 * mm, W - 146 * mm], BLUE),
    p(f"zip_code adds <b>{zd['zip_gain']:+.4f}</b> ROC-AUC; all geography (zip + state) adds {zd['geography_gain']:+.4f}. "
      f"Removing zip_code changes the decision for <b>{zd['decision_flips_without_zip']:,}</b> applicants "
      f"({zd['decision_flips_without_zip'] / zips['loans'].sum():.1%}) under a reference policy that approves the safest 70%:", "b"),
    table(zip_rows, [60 * mm, 18 * mm, 25 * mm, 24 * mm, 26 * mm, W - 153 * mm], GREY),
    p(zip_reading, "small"),
    p("B. Is the model equally accurate across states?", "h"),
    p(f"Fairness criterion: <b>the same PD should mean the same risk everywhere</b>. For the {len(states)} states with ≥ 1,000 validation loans "
      f"({states['loans'].sum() / zips['loans'].sum():.0%} of all loans), the calibration gap (mean PD − actual default rate, out-of-fold) ranges "
      f"from {states['calibration_gap_pts'].min():+.1f} to {states['calibration_gap_pts'].max():+.1f} points; ROC-AUC from "
      f"{states['roc_auc'].min():.3f} to {states['roc_auc'].max():.3f}."),
    Image(str(RESULTS / "f2_state_calibration.png"), width=W, height=W * 0.34),
    p((f"Flagged beyond ±2 points: <b>{', '.join(flagged['state'])}</b>. A positive gap means the state's applicants are charged a higher "
       "PD than their actual risk; these states are candidates for monitoring once the model is live."
       if len(flagged) else "No state is beyond ±2 points."), "small"),
    PageBreak(),
    p("C. Reason codes: why did this applicant get this PD?", "h"),
    p("SHAP values split every applicant's score into contributions per input; they add up exactly to the model's output (checked for all "
      f"162,745 loans). The {choice['n_features']} columns are grouped into concepts (all <i>addr_state_*</i> dummies → \"State\", missing "
      "flags → their feature), and each concept becomes a sentence with the applicant's own value. Up to 4 risk-raising reasons are "
      "returned: the usual number of principal reasons in US adverse-action (decline) notices."),
    Table([[Image(str(CONCEPT_PNG), width=W * 0.5, height=W * 0.5 * 0.64),
            table(reason_rows, [52 * mm, W * 0.48 - 52 * mm], GREEN)]],
          colWidths=[W * 0.52, W * 0.48], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
    p("Worked examples (validation applicants; ▲ raised the risk, ▼ lowered it)", "h"),
    Table([[example_block("low risk"), example_block("medium risk"), example_block("high risk")]],
          colWidths=[W / 3] * 3, style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                                        ("LINEBEFORE", (1, 0), (2, 0), 0.5, colors.HexColor("#d1d5db")),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 5)]),
    p("What we learned", "h"),
    *bullets([
        f"<b>Final model after Stage 2:</b> {choice['chosen_model']}, {choice['n_features']} features, monotonicity violations "
        f"{choice['monotonicity_violations_final']}, out-of-fold ROC-AUC {oof['roc_auc']:.4f}, Brier {oof['brier']:.5f}, mean PD "
        f"{pct(oof['mean_pd'])} vs actual {pct(oof['default_rate'])}.",
        "Reasons are concrete and checkable by the applicant (\"Loan term: 60 months\", \"Credit score (FICO): 660\"), not model jargon.",
        "Target-encoded inputs such as job title are explained as \"historical default rate of similar titles\"; like zip codes, job title "
        "can be a weak proxy for protected groups and is a candidate for the same ablation test later.",
        "Limits: fairness here is checked only through geography because the data has no protected attributes; a real lender would also "
        "run disparate-impact tests on demographic data it holds separately.",
    ]),
    p("Next: Stage 3, the decision layer (risk bands, approve / review / decline thresholds, expected loss).", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/final/02_reason_codes_fairness · see F2_reason_codes_fairness.ipynb (run on Kaggle)")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="Final model - Stage 2").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
