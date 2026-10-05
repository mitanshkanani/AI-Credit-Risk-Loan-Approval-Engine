"""Build models/M1_logistic_regression/M1_logistic_regression.pdf from the saved M1 results.

    python models/M1_logistic_regression/build_m1_pdf.py
"""

import json
from pathlib import Path

import numpy as np
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
ROOT = HERE.parents[1]
OUT = HERE / "M1_logistic_regression.pdf"

summary = json.loads((RESULTS / "m1_metrics.json").read_text(encoding="utf-8"))
tuning = pd.read_csv(RESULTS / "m1_tuning_results.csv")
tuning["class_weight"] = tuning["class_weight"].fillna("None")   # CSV reads the string "None" as missing
coefs = pd.read_csv(RESULTS / "m1_coefficients.csv")
curves = pd.read_csv(RESULTS / "m1_approval_curve.csv")
segments = pd.read_csv(RESULTS / "m1_segment_metrics.csv")
board = pd.read_csv(ROOT / "models" / "leaderboard.csv")
m0 = board[(board.model == "M0") & (board.version == "A") & (board.split == "validation")].iloc[0]
cards, chosen = summary["metrics"], summary["chosen"]
A_val, B_val, A_tr, B_tr = cards["A"]["validation"], cards["B"]["validation"], cards["A"]["train"], cards["B"]["train"]

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
GREY, BLUE, GREEN, ORANGE = (colors.HexColor(c) for c in ("#4b5563", "#3b6fd4", "#2e9b5b", "#d4840f"))
S = {
    "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Body", fontSize=10, leading=14, textColor=MUTED, spaceAfter=7),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=6, spaceAfter=4),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.3, leading=12.8, textColor=INK, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.3, leading=12.8, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=2),
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


def f4(x):
    return f"{x:.4f}"


# ------------------------------------------------------------------ derived facts
best = tuning.groupby(["version", "prep"])["val_roc_auc"].max()
cw = tuning.groupby(["version", "class_weight"])[["val_roc_auc", "val_brier", "val_mean_pd"]].max()
c_range = {}
for v in ("A", "B"):
    sel = tuning[(tuning.version == v) & (tuning.prep == chosen[v]["prep"]) & (tuning.class_weight.astype(str) == "None")]
    c_range[v] = sel.val_roc_auc.max() - sel.val_roc_auc.min()
curve = curves.pivot(index="approval_rate", columns="model", values="bad_rate_among_approved")
seg = {(r.version, int(r.term_months)): r for r in segments.itertuples()}
kept = (B_val["roc_auc"] - 0.5) / (A_val["roc_auc"] - 0.5)
iters = summary["n_iter"]


def coef(v, f):
    return float(coefs[(coefs.version == v) & (coefs.feature == f)]["weight"].iloc[0])


top_b = (coefs[coefs.version == "B"].assign(a=lambda d: d.weight.abs())
         .sort_values("a", ascending=False).head(8))
top_rows = [["Feature (version B)", "Weight", "Odds ratio", "Reading"]]
reading = {
    "term_months": "60-month loans are much riskier",
    "loan_amnt": "bigger loans, higher risk",
    "annual_inc": "higher income, lower risk",
    "fico_range_low": "higher FICO, lower risk",
    "dti": "more debt per income, higher risk",
    "acc_open_past_24mths": "many new accounts, higher risk",
    "mo_sin_old_rev_tl_op": "older revolving history, lower risk",
    "tot_hi_cred_lim": "larger total credit limits, lower risk",
    "bc_open_to_buy": "more unused card credit, lower risk",
    "credit_history_months": "see caveat below",
}
for r in top_b.itertuples():
    top_rows.append([r.feature, f"{r.weight:+.3f}", f"{r.odds_ratio:.2f}", reading.get(r.feature, "")])

story = [
    p("M1 · Logistic Regression", "title"),
    p("Model 2 of the ladder · the explainable baseline · AI Credit Risk / Loan Approval Engine", "sub"),
    callout([p("<b>In one sentence:</b> Logistic Regression gives every feature a weight, adds up <i>weight × value</i> for "
               "an applicant, and turns the total into a probability of default with the sigmoid function: "
               "<b>PD = 1 / (1 + e<super>−(b0 + w1·x1 + … + wn·xn)</super>)</b>. A positive weight raises the PD, a negative one "
               "lowers it, and every decision can be explained feature by feature. That is why banks still use it.")],
            BLUE, "#eef4ff"),
    p("How it was set up", "h"),
    *bullets([
        f"<b>Version A</b>: {summary['n_features']['A']} features (all 190 minus <i>fico_range_high</i>, which is a 1.000 "
        f"correlated copy of fico_range_low). <b>Version B</b>: {summary['n_features']['B']} features, also without LendingClub's "
        "own risk view (grade, sub_grade, int_rate, installment). <b>B is the deployable model.</b>",
        f"<b>Model-specific preprocessing</b> (fitted on train only): <b>log1p</b> on the {len(summary['log1p_columns'])} heavily "
        "skewed, non-negative columns (skew &gt; 2, e.g. annual_inc, revol_bal, tot_coll_amt), then <b>StandardScaler</b> on "
        "everything, so all weights are on the same scale.",
        "<b>Tuning</b>: 20 settings per version, chosen on validation ROC-AUC; ties within 0.0005 go to class_weight=None, "
        "then to the smallest C (the simplest model).",
    ]),
    p("Parameters", "h"),
    table([
        ["Parameter", "Version A", "Version B", "What it means"],
        ["C", f"{chosen['A']['C']}", f"{chosen['B']['C']}", "Inverse regularisation strength. Small C shrinks weights toward 0 "
                                                          "(simpler, more stable model). Tried 0.0001 to 1."],
        ["penalty", "l2", "l2", "Ridge penalty: shrinks all weights smoothly, works well with correlated features."],
        ["class_weight", str(chosen["A"]["class_weight"]), str(chosen["B"]["class_weight"]),
         "\"balanced\" up-weights defaults. It gave the same AUC but wrecked probabilities, so None was kept."],
        ["preprocessing", chosen["A"]["prep"], chosen["B"]["prep"], "log1p on skewed columns + StandardScaler (vs scale only)."],
        ["solver / max_iter", "lbfgs / 3000", "lbfgs / 3000", f"Optimiser; converged in {iters['A']} (A) and {iters['B']} (B) iterations."],
    ], [26 * mm, 20 * mm, 20 * mm, W - 66 * mm]),
    p("What the tuning showed (validation)", "h"),
    table([
        ["Experiment", "Version A", "Version B", "Takeaway"],
        ["Best ROC-AUC: scale only → log1p + scale",
         f"{best[('A', 'scale')]:.4f} → {best[('A', 'log+scale')]:.4f}", f"{best[('B', 'scale')]:.4f} → {best[('B', 'log+scale')]:.4f}",
         "Taming skewed columns helps a little, more for B."],
        ["class_weight None vs balanced: Brier",
         f"{cw.loc[('A', 'None'), 'val_brier']:.4f} vs {cw.loc[('A', 'balanced'), 'val_brier']:.4f}",
         f"{cw.loc[('B', 'None'), 'val_brier']:.4f} vs {cw.loc[('B', 'balanced'), 'val_brier']:.4f}",
         f"Same ranking, but balanced pushes mean PD to about {cw.loc[('A', 'balanced'), 'val_mean_pd']:.0%}: unusable as a probability."],
        ["Effect of C (0.0001 to 1), chosen preprocessing, class_weight None",
         f"AUC range {c_range['A']:.4f}", f"AUC range {c_range['B']:.4f}",
         "Plenty of data, so regularisation barely matters."],
    ], [48 * mm, 30 * mm, 30 * mm, W - 108 * mm]),
    p("Results", "h"),
    table([
        ["Model (split)", "ROC-AUC", "KS", "PR-AUC", "Brier", "Log loss", "Mean PD", "Actual rate"],
        ["M0 (validation)", f4(m0.roc_auc), f4(m0.ks), f4(m0.pr_auc), f4(m0.brier), f4(m0.log_loss), pct(m0.mean_pd), pct(m0.default_rate)],
        ["M1-A (train)", f4(A_tr["roc_auc"]), f4(A_tr["ks"]), f4(A_tr["pr_auc"]), f4(A_tr["brier"]), f4(A_tr["log_loss"]), pct(A_tr["mean_pd"]), pct(A_tr["default_rate"])],
        ["<b>M1-A (validation)</b>", f4(A_val["roc_auc"]), f4(A_val["ks"]), f4(A_val["pr_auc"]), f4(A_val["brier"]), f4(A_val["log_loss"]), pct(A_val["mean_pd"]), pct(A_val["default_rate"])],
        ["M1-B (train)", f4(B_tr["roc_auc"]), f4(B_tr["ks"]), f4(B_tr["pr_auc"]), f4(B_tr["brier"]), f4(B_tr["log_loss"]), pct(B_tr["mean_pd"]), pct(B_tr["default_rate"])],
        ["<b>M1-B (validation)</b>", f4(B_val["roc_auc"]), f4(B_val["ks"]), f4(B_val["pr_auc"]), f4(B_val["brier"]), f4(B_val["log_loss"]), pct(B_val["mean_pd"]), pct(B_val["default_rate"])],
    ], [31 * mm, 19 * mm, 15 * mm, 18 * mm, 16 * mm, 17 * mm, 17 * mm, W - 133 * mm], BLUE),
    p("All sanity checks passed: both solvers converged, both models beat M0 on every metric, and both stay well "
      "below the 0.80 leakage flag.", "small"),
    PageBreak(),
    p("What the model learned", "h"),
    table(top_rows, [44 * mm, 18 * mm, 20 * mm, W - 82 * mm], GREEN),
    p("Weights are per 1 standard deviation (log-odds); odds ratio = e<super>weight</super>. All five intuition checks pass in both "
      "versions: FICO ↓, DTI ↑, 60-month term ↑, income ↓, recent inquiries ↑ (and interest rate ↑ in A).", "small"),
    Spacer(1, 4),
    Table([[Image(str(RESULTS / "m1_approval_curve.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.63),
            Image(str(RESULTS / "m1_calibration.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.85)]],
          colWidths=[W / 2, W / 2], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    p("Key findings", "h"),
    *bullets([
        f"<b>Real ranking power.</b> Approving the safest 50% of validation applicants gives a bad rate of "
        f"{pct(curve.loc[0.5, 'M1-B'])} with M1-B (vs {pct(curve.loc[0.5, 'M0'])} for M0); the safest 10% default at only "
        f"{pct(curve.loc[0.1, 'M1-B'])}.",
        f"<b>LendingClub's grades add little.</b> A beats B by only {A_val['roc_auc'] - B_val['roc_auc']:.4f} ROC-AUC; B keeps "
        f"{kept:.1%} of A's ranking power above a coin flip. Our independent model is close to LendingClub's own underwriting.",
        f"<b>Probabilities run low on newer loans.</b> Mean PD {pct(B_val['mean_pd'])} vs actual {pct(B_val['default_rate'])}; the "
        "riskiest decile is under-predicted most. This is the 2015 drift M0 already showed, so calibration on validation is needed.",
        f"<b>60-month loans are harder.</b> M1-B ROC-AUC is {seg[('B', 36)].roc_auc:.3f} on 36-month vs {seg[('B', 60)].roc_auc:.3f} "
        f"on 60-month loans, and 60-month PDs are too low by {-seg[('B', 60)].calibration_gap * 100:.1f} points.",
        f"<b>Validation AUC is above train AUC</b> ({B_val['roc_auc']:.3f} vs {B_tr['roc_auc']:.3f} for B). This is not leakage: "
        "validation was never used for fitting. Possible reasons: the 2015 H1 population may be easier to rank, and training rows "
        "use noisier out-of-fold target encodings. Worth re-checking with the tree models.",
        f"<b>Caveat on correlated features.</b> credit_history_months gets a positive weight ({coef('B', 'credit_history_months'):+.3f}) "
        f"while the closely related mo_sin_old_rev_tl_op gets a negative one ({coef('B', 'mo_sin_old_rev_tl_op'):+.3f}). Correlated "
        "features split and offset each other, so single weights must be read with care.",
    ]),
    Spacer(1, 3),
    callout([p(f"<b>The bar M2 (Random Forest) must clear on validation, version B:</b> ROC-AUC {B_val['roc_auc']:.4f}, "
               f"KS {B_val['ks']:.4f}, PR-AUC {B_val['pr_auc']:.4f}, Brier {B_val['brier']:.4f}, log loss {B_val['log_loss']:.4f}. "
               "Per the model strategy, a clear gain means non-linear effects matter and boosting (M3/M4) is worth the effort.")],
            ORANGE, "#fff4e5"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/M1_logistic_regression · see M1_logistic_regression.ipynb for the executed code")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=14 * mm,
                  bottomMargin=16 * mm, title="M1 - Logistic Regression").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
