"""Build models/M3_lightgbm/M3_lightgbm.pdf from the saved M3 results.

    python models/M3_lightgbm/build_m3_pdf.py
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
OUT = HERE / "M3_lightgbm.pdf"

summary = json.loads((RESULTS / "m3_metrics.json").read_text(encoding="utf-8"))
tuning = pd.read_csv(RESULTS / "m3_tuning_results.csv")
experiments = pd.read_csv(RESULTS / "m3_experiments.csv")
example = pd.read_csv(RESULTS / "m3_example_explanation.csv")
curves = pd.read_csv(RESULTS / "m3_approval_curve.csv")
segments = pd.read_csv(RESULTS / "m3_segment_metrics.csv")
board = pd.read_csv(ROOT / "models" / "leaderboard.csv")
cards, chosen, final_lr, rounds = summary["metrics"], summary["chosen"], summary["final_learning_rate"], summary["best_iteration"]
env = summary["environment"]


def lb(model, version, split="validation"):
    return board[(board.model == model) & (board.version == version) & (board.split == split)].iloc[0]


m0, m1a, m1b, m2a, m2b = lb("M0", "A"), lb("M1", "A"), lb("M1", "B"), lb("M2", "A"), lb("M2", "B")
A, B, BM, B_tr = (cards["A"]["validation"], cards["B"]["validation"], cards["B-mono"]["validation"], cards["B"]["train"])

# Example applicant headline, read from the executed notebook's printed output
nb = json.loads((HERE / "M3_lightgbm.ipynb").read_text(encoding="utf-8"))
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
gain_b_m1, gain_b_m2 = B["roc_auc"] - m1b.roc_auc, B["roc_auc"] - m2b.roc_auc
gain_a_m1 = A["roc_auc"] - m1a.roc_auc
clear = gain_b_m1 > 0.005
baseline = tuning[tuning.trial == 0].set_index("version")["val_roc_auc"]
best_search = tuning.groupby("version")["val_roc_auc"].max()
search_minutes = tuning["seconds"].sum() / 60
exp = {(r.experiment, r.version): r for r in experiments.itertuples()}
curve = curves.pivot(index="approval_rate", columns="model", values="bad_rate_among_approved")
seg = {(r.model, int(r.term_months)): r for r in segments.itertuples()}
mono_cost = B["roc_auc"] - BM["roc_auc"]

_y = pd.read_parquet(ROOT / "final_preprocessed_data_v2" / "y_validation.parquet")
_m3 = pd.read_parquet(RESULTS / "m3_validation_pd.parquet")
_m1 = pd.read_parquet(ROOT / "models" / "M1_logistic_regression" / "results" / "m1_B_validation_pd.parquet")
assert (_y["id"].astype(str).to_numpy() == _m3["id"].astype(str).to_numpy()).all()
_f = pd.DataFrame({"p": _m3["pd_m3_B"], "y": _y["target"].astype(int)})
_f["bin"] = pd.qcut(_f["p"], 10, labels=False, duplicates="drop")
calib_b = _f.groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean"))
spearman_m1 = _m1["pd_m1_B"].corr(_m3["pd_m3_B"], method="spearman")

verdict = (
    f"<b>Verdict: yes, boosting wins.</b> LightGBM beats Logistic Regression by <b>{gain_b_m1:+.4f}</b> ROC-AUC on the deployable "
    f"version B (above the +0.005 'clear gain' bar) and Random Forest by {gain_b_m2:+.4f}, with a better Brier score "
    f"({m1b.brier:.4f} → {B['brier']:.4f}). Non-linear effects do matter once trees correct each other's mistakes. "
    "M3 is the leading candidate."
    if clear else
    f"<b>Verdict: no clear gain.</b> LightGBM changes ROC-AUC by {gain_b_m1:+.4f} on version B versus Logistic Regression "
    "(bar: +0.005). Logistic Regression stays the simpler, explainable choice."
)

param_rows = [["Parameter", "Version A", "Version B", "What it means"]]
meaning = {
    "num_leaves": "Leaves per tree (tree complexity). Searched 15-255.",
    "min_data_in_leaf": "Minimum loans per leaf (regularisation). Searched 20-2,000.",
    "feature_fraction": "Share of features each tree may use. Searched 0.4-1.0.",
    "bagging_fraction": "Share of loans each tree trains on. Searched 0.5-1.0.",
    "lambda_l1": "L1 penalty on leaf values. Searched 0.0001-10.",
    "lambda_l2": "L2 penalty on leaf values. Searched 0.0001-10.",
    "min_gain_to_split": "Minimum loss improvement to make a split. Searched 0-0.5.",
}
for k, text in meaning.items():
    param_rows.append([k, fmt(chosen["A"][k]), fmt(chosen["B"][k]), text])
param_rows.append(["learning_rate / rounds", f"{final_lr['A']} / {rounds['A']}", f"{final_lr['B']} / {rounds['B']}",
                   "Step size and the number of trees chosen by early stopping (100-round patience)."])

story = [
    p("M3 · LightGBM (gradient boosting)", "title"),
    p("Model 4 of the ladder · the main candidate · AI Credit Risk / Loan Approval Engine", "sub"),
    callout([p("<b>In one sentence:</b> boosting builds trees <b>one after another</b>. Each new small tree is fitted to the "
               "mistakes that all previous trees together still make, and is added with a small weight (the learning rate). The sum "
               "of all trees is the log-odds of default, which the sigmoid turns into a PD. Random Forest averages independent deep "
               "trees; boosting uses many shallow trees that <b>cooperate</b>.")], BLUE, "#eef4ff"),
    p("How it was set up", "h"),
    *bullets([
        f"Same feature sets as M1/M2: <b>A</b> {summary['n_features']['A']}, <b>B</b> {summary['n_features']['B']} features "
        "(no LendingClub grades; deployable). No scaling needed.",
        f"<b>Run on Kaggle</b> (private CPU notebook, {env['cpu_cores']} cores, LightGBM {env['lightgbm']}), driven from the CLI; "
        "only train and validation were uploaded, so the test set stays sealed.",
        f"<b>Random search</b>: the same {summary['search']['settings_per_version']} settings for A and B at learning rate "
        f"{summary['search']['learning_rate']} ({search_minutes:.0f} minutes). Each fit stops when validation AUC has not improved for "
        "100 rounds. Validation therefore both stops and selects, so its score is slightly optimistic; the sealed test set "
        "will give the honest number.",
    ]),
    p("Chosen parameters", "h"),
    table(param_rows, [34 * mm, 24 * mm, 24 * mm, W - 82 * mm]),
    p("Experiments (validation)", "h"),
    table([
        ["Experiment", "Version A", "Version B", "Takeaway"],
        ["Untuned baseline → best of search (ROC-AUC)", f"{baseline['A']:.4f} → {best_search['A']:.4f}",
         f"{baseline['B']:.4f} → {best_search['B']:.4f}", "Tuning adds a little; the model family does the heavy lifting."],
        ["Learning rate 0.05 vs 0.02 (ROC-AUC)", f"{exp[('chosen, lr 0.05', 'A')].val_roc_auc:.4f} vs {exp[('chosen, lr 0.02', 'A')].val_roc_auc:.4f}",
         f"{exp[('chosen, lr 0.05', 'B')].val_roc_auc:.4f} vs {exp[('chosen, lr 0.02', 'B')].val_roc_auc:.4f}",
         f"Kept: {final_lr['A']} (A), {final_lr['B']} (B); 0.02 only if +0.0005."],
        ["is_unbalance=True: ROC-AUC / Brier", f"{exp[('is_unbalance=True', 'A')].val_roc_auc:.4f} / {exp[('is_unbalance=True', 'A')].val_brier:.4f}",
         f"{exp[('is_unbalance=True', 'B')].val_roc_auc:.4f} / {exp[('is_unbalance=True', 'B')].val_brier:.4f}",
         "Up-weighting defaults ruins probabilities again; not used."],
        ["Monotonic constraints (B): ROC-AUC", "-", f"{BM['roc_auc']:.4f} ({-mono_cost:+.4f})",
         "FICO ↓, DTI ↑, income ↓, 60-month ↑, inquiries ↑, utilisation ↑ risk."],
    ], [52 * mm, 32 * mm, 32 * mm, W - 116 * mm]),
    p("Results (validation unless marked)", "h"),
    table([
        ["Model", "ROC-AUC", "KS", "PR-AUC", "Brier", "Log loss", "Mean PD", "Actual"],
        ["M0", f4(m0.roc_auc), f4(m0.ks), f4(m0.pr_auc), f4(m0.brier), f4(m0.log_loss), pct(m0.mean_pd), pct(m0.default_rate)],
        ["M1-A / M2-A", f"{m1a.roc_auc:.4f} / {m2a.roc_auc:.4f}", f4(m1a.ks), f4(m1a.pr_auc), f4(m1a.brier), f4(m1a.log_loss), pct(m1a.mean_pd), pct(m1a.default_rate)],
        ["<b>M3-A</b>", f4(A["roc_auc"]), f4(A["ks"]), f4(A["pr_auc"]), f4(A["brier"]), f4(A["log_loss"]), pct(A["mean_pd"]), pct(A["default_rate"])],
        ["M1-B / M2-B", f"{m1b.roc_auc:.4f} / {m2b.roc_auc:.4f}", f4(m1b.ks), f4(m1b.pr_auc), f4(m1b.brier), f4(m1b.log_loss), pct(m1b.mean_pd), pct(m1b.default_rate)],
        ["<b>M3-B</b>", f4(B["roc_auc"]), f4(B["ks"]), f4(B["pr_auc"]), f4(B["brier"]), f4(B["log_loss"]), pct(B["mean_pd"]), pct(B["default_rate"])],
        ["M3-B-mono", f4(BM["roc_auc"]), f4(BM["ks"]), f4(BM["pr_auc"]), f4(BM["brier"]), f4(BM["log_loss"]), pct(BM["mean_pd"]), pct(BM["default_rate"])],
        ["M3-B (train)", f4(B_tr["roc_auc"]), f4(B_tr["ks"]), f4(B_tr["pr_auc"]), f4(B_tr["brier"]), f4(B_tr["log_loss"]), pct(B_tr["mean_pd"]), pct(B_tr["default_rate"])],
    ], [27 * mm, 26 * mm, 15 * mm, 17 * mm, 16 * mm, 17 * mm, 17 * mm, W - 135 * mm], BLUE),
    p("M1/M2 rows show ROC-AUC for both; the other M1/M2 columns are M1's. All sanity checks passed (early stopping triggered, beats M0, "
      "below the 0.80 leakage flag).", "small"),
    PageBreak(),
    callout([p(verdict)], GREEN if clear else ORANGE, "#ecf8f0" if clear else "#fff4e5"),
    Spacer(1, 3),
    Table([[Image(str(RESULTS / "m3_learning_curve.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.6),
            Image(str(RESULTS / "m3_calibration.png"), width=W / 2 - 14 * mm, height=(W / 2 - 14 * mm) * 0.85)]],
          colWidths=[W / 2, W / 2], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    Table([[Image(str(RESULTS / "m3_shap.png"), width=W * 0.47, height=W * 0.47 * 0.77),
            [p(f"<b>Why one risky applicant got PD {ex_pd}%</b> (average {ex_avg}%; actually defaulted: {ex_default})", "c"),
             Spacer(1, 3),
             table([["Feature", "Value", "SHAP (log-odds)"]] +
                   [[r.feature, f"{r.value:,.4g}", f"{r.shap_log_odds:+.3f}"] for r in example.head(7).itertuples()],
                   [34 * mm, 22 * mm, 24 * mm], GREEN),
             Spacer(1, 3),
             p("Positive = pushed risk up. These per-applicant contributions are what the decision layer will turn into reason codes.", "small")]]],
          colWidths=[W * 0.5, W * 0.5], style=[("VALIGN", (0, 0), (-1, -1), "TOP")]),
    p("Key findings", "h"),
    *bullets([
        f"<b>Ranking:</b> version B ROC-AUC {m1b.roc_auc:.4f} (M1) → {m2b.roc_auc:.4f} (M2) → <b>{B['roc_auc']:.4f}</b> (M3); version A "
        f"{m1a.roc_auc:.4f} → {A['roc_auc']:.4f} ({gain_a_m1:+.4f}). Approving the safest 50% gives a bad rate of {pct(curve.loc[0.5, 'M3-B'])} "
        f"with M3-B vs {pct(curve.loc[0.5, 'M1-B'])} with M1-B (M0: {pct(curve.loc[0.5, 'M0'])}).",
        f"<b>Monotonic constraints cost {mono_cost:+.4f} ROC-AUC</b> on B. The constrained model can never say \"higher FICO = riskier\", "
        "which makes it easier to defend to a credit committee or regulator; it is a strong option for deployment.",
        (f"<b>{'Little' if B_tr['roc_auc'] - B['roc_auc'] < 0.03 else 'Noticeable'} overfitting:</b> train ROC-AUC "
         f"{B_tr['roc_auc']:.4f} vs validation {B['roc_auc']:.4f} for B (Random Forest: {lb('M2', 'B', 'train').roc_auc:.3f} vs "
         f"{m2b.roc_auc:.3f}). Early stopping on validation limits how far the trees can memorise the training loans."),
        f"<b>Calibration:</b> mean PD {pct(B['mean_pd'])} vs actual {pct(B['default_rate'])}; the riskiest decile is predicted at "
        f"{calib_b.iloc[-1].predicted:.1%} vs {calib_b.iloc[-1].actual:.1%} actual. Same 2015 drift as before, so calibration on validation "
        "is still a required step before deployment.",
        f"<b>LendingClub's grades:</b> A beats B by {A['roc_auc'] - B['roc_auc']:.4f} ROC-AUC. <b>Segments (B):</b> ROC-AUC "
        f"{seg[('B', 36)].roc_auc:.3f} (36-month) vs {seg[('B', 60)].roc_auc:.3f} (60-month). Rank agreement with M1-B: Spearman {spearman_m1:.3f}.",
    ]),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/M3_lightgbm · see M3_lightgbm.ipynb for the executed code (run on Kaggle)")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="M3 - LightGBM").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
