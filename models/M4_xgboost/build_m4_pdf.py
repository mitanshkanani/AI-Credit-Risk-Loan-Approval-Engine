"""Build models/M4_xgboost/M4_xgboost.pdf from the saved M4 results.

    python models/M4_xgboost/build_m4_pdf.py
"""

import json
import re
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
ROOT = HERE.parents[1]
OUT = HERE / "M4_xgboost.pdf"

summary = json.loads((RESULTS / "m4_metrics.json").read_text(encoding="utf-8"))
tuning = pd.read_csv(RESULTS / "m4_tuning_results.csv")
m3_tuning = pd.read_csv(ROOT / "models" / "M3_lightgbm" / "results" / "m3_tuning_results.csv")
experiments = pd.read_csv(RESULTS / "m4_experiments.csv")
h2h = pd.read_csv(RESULTS / "m4_vs_m3.csv").set_index("version")
shap_imp = pd.read_csv(RESULTS / "m4_shap_importance.csv")
m3_shap = pd.read_csv(ROOT / "models" / "M3_lightgbm" / "results" / "m3_shap_importance.csv")
example = pd.read_csv(RESULTS / "m4_example_explanation.csv")
curves = pd.read_csv(RESULTS / "m4_approval_curve.csv")
segments = pd.read_csv(RESULTS / "m4_segment_metrics.csv")
board = pd.read_csv(ROOT / "models" / "leaderboard.csv")
cards, chosen, final_lr, rounds = summary["metrics"], summary["chosen"], summary["final_learning_rate"], summary["best_iteration"]
env = summary["environment"]


def lb(model, version, split="validation"):
    return board[(board.model == model) & (board.version == version) & (board.split == split)].iloc[0]


m1a, m1b, m3a, m3b, m3bm = lb("M1", "A"), lb("M1", "B"), lb("M3", "A"), lb("M3", "B"), lb("M3", "B-mono")
A, B, BM, B_tr = (cards["A"]["validation"], cards["B"]["validation"], cards["B-mono"]["validation"], cards["B"]["train"])

nb = json.loads((HERE / "M4_xgboost.ipynb").read_text(encoding="utf-8"))
printed = "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", [])
                  if o.get("output_type") == "stream")
m = re.search(r"Example applicant: PD ([\d.]+)% \(average validation PD ([\d.]+)%\); actually defaulted: (\w+)", printed)
ex_pd, ex_avg, ex_default = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")

FONTS = Path("C:/Windows/Fonts")
pdfmetrics.registerFont(TTFont("Body", str(FONTS / "arial.ttf")))
pdfmetrics.registerFont(TTFont("Body-Bold", str(FONTS / "arialbd.ttf")))
pdfmetrics.registerFont(TTFont("Body-Italic", str(FONTS / "ariali.ttf")))
pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body-Italic", boldItalic="Body-Bold")

INK, MUTED = colors.HexColor("#1f2937"), colors.HexColor("#6b7280")
GREY, BLUE, GREEN, ORANGE = (colors.HexColor(c) for c in ("#4b5563", "#3b6fd4", "#2e9b5b", "#d4840f"))
S = {
    "title": ParagraphStyle("t", fontName="Body-Bold", fontSize=20, leading=24, textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName="Body", fontSize=10, leading=14, textColor=MUTED, spaceAfter=6),
    "h": ParagraphStyle("h", fontName="Body-Bold", fontSize=12.5, leading=16, textColor=INK, spaceBefore=5, spaceAfter=3),
    "b": ParagraphStyle("b", fontName="Body", fontSize=9.2, leading=12.6, textColor=INK, spaceAfter=3),
    "li": ParagraphStyle("li", fontName="Body", fontSize=9.1, leading=12.4, textColor=INK, leftIndent=11, bulletIndent=1, spaceAfter=1.8),
    "c": ParagraphStyle("c", fontName="Body", fontSize=8.4, leading=11, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-Bold", fontSize=8.5, leading=11, textColor=colors.white),
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
             ("TOPPADDING", (0, 0), (-1, -1), 2.4), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.4)]
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


def fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


# ------------------------------------------------------------------ derived facts
diff_b = B["roc_auc"] - m3b.roc_auc
baseline = tuning[tuning.trial == 0].set_index("version")["val_roc_auc"]
best_search = tuning.groupby("version")["val_roc_auc"].max()
search_minutes = tuning["seconds"].sum() / 60
sec_per_fit = {"M3": m3_tuning["seconds"].mean(), "M4": tuning["seconds"].mean()}
exp = {(r.experiment, r.version): r for r in experiments.itertuples()}
curve = curves.pivot(index="approval_rate", columns="model", values="bad_rate_among_approved")
seg = {(r.model, int(r.term_months)): r for r in segments.itertuples()}
mono_cost = B["roc_auc"] - BM["roc_auc"]
top10_overlap = len(set(shap_imp.head(10)["feature"]) & set(m3_shap.head(10)["feature"]))

if diff_b > 0.005:
    verdict, vcolor = (f"<b>Verdict: XGBoost wins.</b> It beats LightGBM by {diff_b:+.4f} ROC-AUC on version B, more than the "
                       "0.005 tie margin, so M4 becomes the main candidate."), GREEN
elif diff_b < -0.005:
    verdict, vcolor = (f"<b>Verdict: LightGBM stays.</b> XGBoost is {diff_b:+.4f} ROC-AUC behind LightGBM on version B."), ORANGE
else:
    faster = "LightGBM" if sec_per_fit["M3"] <= sec_per_fit["M4"] else "XGBoost"
    verdict, vcolor = (f"<b>Verdict: a tie that confirms the result.</b> XGBoost and LightGBM are within the 0.005 tie margin on "
                       f"version B (M4 − M3 = {diff_b:+.4f} ROC-AUC) and rank applicants almost identically (Spearman "
                       f"{h2h.loc['B', 'Spearman(M3, M4)']:.3f}). The boosting gain over Logistic Regression is real, not a quirk "
                       f"of one library. By the strategy's rule (keep the faster / simpler one in a tie), <b>{faster}</b> is preferred "
                       f"on speed: about {sec_per_fit['M3']:.0f} s vs {sec_per_fit['M4']:.0f} s per search fit on the same "
                       "Kaggle CPU."), GREEN

param_rows = [["Parameter", "Version A", "Version B", "What it means"]]
meaning = {
    "max_depth": "Maximum tree depth (trees grow level by level). Searched 3-10.",
    "min_child_weight": "Minimum hessian sum per leaf, about loans × 0.14. Searched 1-200.",
    "subsample": "Share of loans each tree trains on. Searched 0.5-1.0.",
    "colsample_bytree": "Share of features each tree may use. Searched 0.4-1.0.",
    "reg_lambda": "L2 penalty on leaf weights. Searched 0.001-10.",
    "reg_alpha": "L1 penalty on leaf weights. Searched 0.0001-10.",
    "gamma": "Minimum loss reduction needed to split. Searched 0-1.",
}
for k, text in meaning.items():
    param_rows.append([k, fmt(chosen["A"][k]), fmt(chosen["B"][k]), text])
param_rows.append(["learning_rate / rounds", f"{final_lr['A']} / {rounds['A']}", f"{final_lr['B']} / {rounds['B']}",
                   "Step size and number of trees chosen by early stopping on validation AUC (100-round patience)."])

story = [
    p("M4 · XGBoost (gradient boosting challenger)", "title"),
    p("Model 5 of the ladder · is LightGBM's result real? · AI Credit Risk / Loan Approval Engine", "sub"),
    callout([p("<b>In one sentence:</b> XGBoost is the same idea as LightGBM (shallow trees built one after another, each fixing the "
               "mistakes of all trees before it) with a different engine: trees grow <b>level by level</b> up to <i>max_depth</i>, where "
               "LightGBM grows them leaf by leaf. If two independent libraries agree, the boosting gain is real.")], BLUE, "#eef4ff"),
    p("How it was set up", "h"),
    *bullets([
        f"Same feature sets as M1-M3: <b>A</b> {summary['n_features']['A']}, <b>B</b> {summary['n_features']['B']} features "
        "(no LendingClub grades; deployable).",
        f"<b>Run on Kaggle only</b> (private CPU notebook, {env['cpu_cores']} cores, XGBoost {env['xgboost']}, <i>tree_method=hist</i>). "
        "XGBoost is not installed locally, so a 3-setting smoke test also ran on Kaggle before the full run. Test set still sealed.",
        f"<b>Random search</b>: the same {summary['search']['settings_per_version']} settings for A and B at learning rate "
        f"{summary['search']['learning_rate']} ({search_minutes:.0f} minutes); early stopping on validation AUC after 100 rounds.",
    ]),
    p("Chosen parameters", "h"),
    table(param_rows, [34 * mm, 24 * mm, 24 * mm, W - 82 * mm]),
    p("Experiments (validation)", "h"),
    table([
        ["Experiment", "Version A", "Version B", "Takeaway"],
        ["Default-style baseline → best of search (ROC-AUC)", f"{baseline['A']:.4f} → {best_search['A']:.4f}",
         f"{baseline['B']:.4f} → {best_search['B']:.4f}",
         f"Tuning adds {best_search['B'] - baseline['B']:+.4f} on B (LightGBM: "
         f"{m3_tuning.groupby('version')['val_roc_auc'].max()['B'] - m3_tuning[m3_tuning.trial == 0].set_index('version')['val_roc_auc']['B']:+.4f})."],
        ["Learning rate 0.05 vs 0.02 (ROC-AUC)", f"{exp[('chosen, lr 0.05', 'A')].val_roc_auc:.4f} vs {exp[('chosen, lr 0.02', 'A')].val_roc_auc:.4f}",
         f"{exp[('chosen, lr 0.05', 'B')].val_roc_auc:.4f} vs {exp[('chosen, lr 0.02', 'B')].val_roc_auc:.4f}",
         f"Kept: {final_lr['A']} (A), {final_lr['B']} (B); 0.02 only if +0.0005."],
        ["scale_pos_weight: ROC-AUC / Brier", f"{exp[('scale_pos_weight', 'A')].val_roc_auc:.4f} / {exp[('scale_pos_weight', 'A')].val_brier:.4f}",
         f"{exp[('scale_pos_weight', 'B')].val_roc_auc:.4f} / {exp[('scale_pos_weight', 'B')].val_brier:.4f}",
         "Up-weighting defaults again ruins probabilities; not used."],
        ["Monotonic constraints (B): ROC-AUC", "-", f"{BM['roc_auc']:.4f} ({-mono_cost:+.4f})",
         "FICO ↓, DTI ↑, income ↓, 60-month ↑, inquiries ↑, utilisation ↑ risk."],
    ], [52 * mm, 32 * mm, 32 * mm, W - 116 * mm]),
    p("Head-to-head with M3 LightGBM (validation)", "h"),
    table([
        ["Version", "M3 ROC-AUC", "M4 ROC-AUC", "M4 − M3", "M3 Brier", "M4 Brier", "Spearman", "Avg of both (info)"],
        *[[v, f4(h2h.loc[v, "M3 ROC-AUC"]), f4(h2h.loc[v, "M4 ROC-AUC"]), f"{h2h.loc[v, 'M4 - M3']:+.4f}",
           f4(h2h.loc[v, "M3 Brier"]), f4(h2h.loc[v, "M4 Brier"]), f"{h2h.loc[v, 'Spearman(M3, M4)']:.3f}",
           f4(h2h.loc[v, "average of M3+M4 ROC-AUC (info)"])] for v in ("A", "B", "B-mono")],
    ], [16 * mm, 20 * mm, 20 * mm, 17 * mm, 17 * mm, 17 * mm, 20 * mm, W - 127 * mm], BLUE),
    p(f"Speed on the same 4-core Kaggle CPU: LightGBM {sec_per_fit['M3']:.0f} s vs XGBoost {sec_per_fit['M4']:.0f} s per search fit "
      f"(average). \"Avg of both\" averages the two models' PDs; shown for information only.", "small"),
    PageBreak(),
    callout([p(verdict)], vcolor, "#ecf8f0" if vcolor == GREEN else "#fff4e5"),
    Spacer(1, 3),
    Table([[Image(str(RESULTS / "m4_learning_curve.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.6),
            Image(str(RESULTS / "m4_calibration.png"), width=W / 2 - 14 * mm, height=(W / 2 - 14 * mm) * 0.85)]],
          colWidths=[W / 2, W / 2], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    Table([[Image(str(RESULTS / "m4_shap.png"), width=W * 0.47, height=W * 0.47 * 0.77),
            [p(f"<b>Why one risky applicant got PD {ex_pd}%</b> (average {ex_avg}%; actually defaulted: {ex_default})", "c"),
             Spacer(1, 3),
             table([["Feature", "Value", "SHAP (log-odds)"]] +
                   [[r.feature, f"{r.value:,.4g}", f"{r.shap_log_odds:+.3f}"] for r in example.head(7).itertuples()],
                   [34 * mm, 22 * mm, 24 * mm], GREEN),
             Spacer(1, 3),
             p(f"Top-10 SHAP features shared with M3: <b>{top10_overlap} / 10</b>. Both libraries rely on the same drivers.", "small")]]],
          colWidths=[W * 0.5, W * 0.5], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
    p("Key findings", "h"),
    *bullets([
        f"<b>Version B ROC-AUC:</b> M1 {m1b.roc_auc:.4f} → M3 {m3b.roc_auc:.4f} → M4 <b>{B['roc_auc']:.4f}</b>; Brier M3 {m3b.brier:.4f} vs "
        f"M4 {B['brier']:.4f}. Version A: M3 {m3a.roc_auc:.4f} vs M4 {A['roc_auc']:.4f}.",
        f"<b>Approval view (B):</b> approving the safest 50% gives a bad rate of {pct(curve.loc[0.5, 'M4-B'])} (M4) vs "
        f"{pct(curve.loc[0.5, 'M3-B'])} (M3) and {pct(curve.loc[0.5, 'M1-B'])} (M1).",
        f"<b>Monotonic constraints</b> cost {mono_cost:+.4f} ROC-AUC with XGBoost (M3: {m3b.roc_auc - m3bm.roc_auc:+.4f}): "
        "nearly free in both libraries.",
        f"<b>Overfitting:</b> train ROC-AUC {B_tr['roc_auc']:.4f} vs validation {B['roc_auc']:.4f} for B. <b>Calibration:</b> mean PD "
        f"{pct(B['mean_pd'])} vs actual {pct(B['default_rate'])}, so the final model still needs calibrating on validation.",
        f"<b>Segments (B):</b> ROC-AUC {seg[('B', 36)].roc_auc:.3f} (36-month) vs {seg[('B', 60)].roc_auc:.3f} (60-month). "
        f"<b>LendingClub's grades:</b> A beats B by {A['roc_auc'] - B['roc_auc']:.4f} ROC-AUC.",
    ]),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/M4_xgboost · see M4_xgboost.ipynb for the executed code (run on Kaggle)")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="M4 - XGBoost").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
