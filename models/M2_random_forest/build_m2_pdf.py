"""Build models/M2_random_forest/M2_random_forest.pdf from the saved M2 results.

    python models/M2_random_forest/build_m2_pdf.py
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
ROOT = HERE.parents[1]
OUT = HERE / "M2_random_forest.pdf"

summary = json.loads((RESULTS / "m2_metrics.json").read_text(encoding="utf-8"))
tuning = pd.read_csv(RESULTS / "m2_tuning_results.csv", keep_default_na=False)
tree_curve = pd.read_csv(RESULTS / "m2_tree_curve.csv")
importance = pd.read_csv(RESULTS / "m2_feature_importance.csv")
curves = pd.read_csv(RESULTS / "m2_approval_curve.csv")
segments = pd.read_csv(RESULTS / "m2_segment_metrics.csv")
board = pd.read_csv(ROOT / "models" / "leaderboard.csv")
cards, chosen, info = summary["metrics"], summary["chosen"], summary["model_info"]


def lb(model, version, split="validation"):
    return board[(board.model == model) & (board.version == version) & (board.split == split)].iloc[0]


m0, m1a, m1b = lb("M0", "A"), lb("M1", "A"), lb("M1", "B")
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
gain_a, gain_b = A_val["roc_auc"] - m1a.roc_auc, B_val["roc_auc"] - m1b.roc_auc
clear_gain = gain_b > 0.005
curve = curves.pivot(index="approval_rate", columns="model", values="bad_rate_among_approved")
seg = {(r.version, int(r.term_months)): r for r in segments.itertuples()}
tc = tree_curve.pivot(index="n_trees", columns="version", values="val_roc_auc")
imp_b = importance[importance.version == "B"].head(20)
perm_top = imp_b.sort_values("permutation_auc_drop", ascending=False).head(6)
leaf_effect = tuning[tuning.class_weight == "None"].groupby(["version", "min_samples_leaf"])["val_roc_auc"].max().unstack()
mf_effect = tuning[tuning.class_weight == "None"].groupby(["version", "max_features"])["val_roc_auc"].max().unstack()
cw_rows = {v: tuning[(tuning.version == v) & (tuning.class_weight == "balanced_subsample")].iloc[0] for v in ("A", "B")}
cw_base = {v: tuning[(tuning.version == v) & (tuning.class_weight == "None")
                     & (tuning.min_samples_leaf == cw_rows[v].min_samples_leaf)
                     & (tuning.max_features == cw_rows[v].max_features)].iloc[0] for v in ("A", "B")}
minutes = tuning["fit_seconds"].sum() / 60

# Decile calibration of M2-B and rank agreement with M1-B, recomputed from the saved validation predictions
_y = pd.read_parquet(ROOT / "final_preprocessed_data_v2" / "y_validation.parquet")
_m2 = pd.read_parquet(RESULTS / "m2_validation_pd.parquet")
_m1 = pd.read_parquet(ROOT / "models" / "M1_logistic_regression" / "results" / "m1_B_validation_pd.parquet")
assert (_y["id"].astype(str).to_numpy() == _m2["id"].to_numpy()).all() and (_m1["id"].to_numpy() == _m2["id"].to_numpy()).all()
_frame = pd.DataFrame({"p": _m2["pd_m2_B"], "y": _y["target"].astype(int)})
_frame["bin"] = pd.qcut(_frame["p"], 10, labels=False, duplicates="drop")
calib_b = _frame.groupby("bin").agg(predicted=("p", "mean"), actual=("y", "mean"))
spearman = _m1["pd_m1_B"].corr(_m2["pd_m2_B"], method="spearman")

verdict = (
    f"<b>Verdict: yes.</b> Random Forest beats Logistic Regression by {gain_b:+.4f} ROC-AUC on version B, a clear gain "
    "(more than 0.005). Non-linear effects and interactions matter, so boosting (M3/M4) is well worth trying."
    if clear_gain else
    f"<b>Verdict: no clear gain.</b> Random Forest changes ROC-AUC by {gain_b:+.4f} on version B versus Logistic Regression "
    "(the bar for a 'clear gain' is +0.005). Averaging independent trees finds little signal that the linear model missed. "
    "This does not rule out non-linearity: gradient boosting (M3), where each tree corrects the previous trees' mistakes, "
    "is the real test, and M1 remains a strong, explainable candidate."
)

story = [
    p("M2 · Random Forest", "title"),
    p("Model 3 of the ladder · does non-linearity matter? · AI Credit Risk / Loan Approval Engine", "sub"),
    callout([p("<b>In one sentence:</b> a Random Forest grows hundreds of decision trees, each on a random sample of loans and "
               "looking at a random subset of features at every split. A leaf's PD is the default rate of the training loans that "
               "landed in it, and the forest's PD is the <b>average over all trees</b>. Averaging many different, noisy trees cancels "
               "their individual mistakes, and the trees find interactions like \"high DTI only matters when income is low\" on their own.")],
            BLUE, "#eef4ff"),
    p("How it was set up", "h"),
    *bullets([
        f"Same feature sets as M1 for a fair fight: <b>A</b> {summary['n_features']['A']} features, <b>B</b> "
        f"{summary['n_features']['B']} (no LendingClub grades; deployable). Trees need <b>no scaling or log transforms</b>.",
        f"<b>Run on Kaggle</b> (private CPU notebook, {summary['environment']['cpu_cores']} cores, about 30 GB RAM, scikit-learn "
        f"{summary['environment']['scikit_learn']}), driven from the Kaggle CLI, after the laptop run hit its memory limit. Only "
        "train and validation data were uploaded, so the test set stayed sealed. Each tree trains on a random half of the loans "
        "(<i>max_samples=0.5</i>).",
        f"Tuning: 7 settings per version at 200 trees ({minutes:.0f} minutes in total), chosen on validation ROC-AUC; then the "
        "winner was refit with 500 trees.",
    ]),
    p("Parameters", "h"),
    table([
        ["Parameter", "Version A", "Version B", "What it means"],
        ["n_estimators", "500", "500", "Number of trees (200 during tuning). More trees = more stable average, never more overfitting."],
        ["min_samples_leaf", str(chosen["A"]["min_samples_leaf"]), str(chosen["B"]["min_samples_leaf"]),
         "Minimum loans per leaf. Larger = smoother, more reliable leaf default rates. Tried 20 / 50 / 100."],
        ["max_features", str(chosen["A"]["max_features"]), str(chosen["B"]["max_features"]),
         "Features considered at each split (sqrt ≈ 12-14). Fewer = more diverse trees. Tried sqrt and 0.3."],
        ["class_weight", str(chosen["A"]["class_weight"]), str(chosen["B"]["class_weight"]), "Up-weighting defaults was checked and not kept."],
        ["max_samples, bootstrap, criterion, random_state", "0.5, True, gini, 42", "0.5, True, gini, 42",
         "Each tree sees a random half of the loans; gini split quality; fixed seed for reproducibility."],
        ["Tree size (average)", f"{info['A']['avg_nodes_per_tree']:,} nodes, depth {info['A']['avg_depth']:.0f}",
         f"{info['B']['avg_nodes_per_tree']:,} nodes, depth {info['B']['avg_depth']:.0f}",
         f"Final fit time: {info['A']['fit_seconds'] / 60:.1f} min (A), {info['B']['fit_seconds'] / 60:.1f} min (B)."],
    ], [32 * mm, 30 * mm, 30 * mm, W - 92 * mm]),
    p("What the tuning showed (validation ROC-AUC, 200 trees)", "h"),
    table([
        ["Experiment", "Version A", "Version B"],
        ["min_samples_leaf 20 / 50 / 100 (best of max_features)",
         " / ".join(f4(leaf_effect.loc["A", k]) for k in (20, 50, 100)),
         " / ".join(f4(leaf_effect.loc["B", k]) for k in (20, 50, 100))],
        ["max_features sqrt / 0.3 (best of leaf sizes)",
         f"{f4(mf_effect.loc['A', 'sqrt'])} / {f4(mf_effect.loc['A', '0.3'])}", f"{f4(mf_effect.loc['B', 'sqrt'])} / {f4(mf_effect.loc['B', '0.3'])}"],
        ["class_weight None vs balanced_subsample: AUC",
         f"{f4(cw_base['A'].val_roc_auc)} vs {f4(cw_rows['A'].val_roc_auc)}", f"{f4(cw_base['B'].val_roc_auc)} vs {f4(cw_rows['B'].val_roc_auc)}"],
        ["... and Brier (lower is better)",
         f"{f4(cw_base['A'].val_brier)} vs {f4(cw_rows['A'].val_brier)}", f"{f4(cw_base['B'].val_brier)} vs {f4(cw_rows['B'].val_brier)}"],
        ["Trees 25 → 100 → 500 (final model)",
         f"{f4(tc.loc[25, 'A'])} → {f4(tc.loc[100, 'A'])} → {f4(tc.loc[500, 'A'])}", f"{f4(tc.loc[25, 'B'])} → {f4(tc.loc[100, 'B'])} → {f4(tc.loc[500, 'B'])}"],
    ], [72 * mm, (W - 72 * mm) / 2, (W - 72 * mm) / 2]),
    p("Results (validation unless marked)", "h"),
    table([
        ["Model", "ROC-AUC", "KS", "PR-AUC", "Brier", "Log loss", "Mean PD", "Actual rate"],
        ["M0", f4(m0.roc_auc), f4(m0.ks), f4(m0.pr_auc), f4(m0.brier), f4(m0.log_loss), pct(m0.mean_pd), pct(m0.default_rate)],
        ["M1-A", f4(m1a.roc_auc), f4(m1a.ks), f4(m1a.pr_auc), f4(m1a.brier), f4(m1a.log_loss), pct(m1a.mean_pd), pct(m1a.default_rate)],
        ["<b>M2-A</b>", f4(A_val["roc_auc"]), f4(A_val["ks"]), f4(A_val["pr_auc"]), f4(A_val["brier"]), f4(A_val["log_loss"]), pct(A_val["mean_pd"]), pct(A_val["default_rate"])],
        ["M1-B", f4(m1b.roc_auc), f4(m1b.ks), f4(m1b.pr_auc), f4(m1b.brier), f4(m1b.log_loss), pct(m1b.mean_pd), pct(m1b.default_rate)],
        ["<b>M2-B</b>", f4(B_val["roc_auc"]), f4(B_val["ks"]), f4(B_val["pr_auc"]), f4(B_val["brier"]), f4(B_val["log_loss"]), pct(B_val["mean_pd"]), pct(B_val["default_rate"])],
        ["M2-B (train)", f4(B_tr["roc_auc"]), f4(B_tr["ks"]), f4(B_tr["pr_auc"]), f4(B_tr["brier"]), f4(B_tr["log_loss"]), pct(B_tr["mean_pd"]), pct(B_tr["default_rate"])],
    ], [27 * mm, 19 * mm, 15 * mm, 18 * mm, 16 * mm, 17 * mm, 17 * mm, W - 129 * mm], BLUE),
    p("The train score is optimistic by design: each training loan was inside the sample of about half the trees. "
      "Validation is the honest number.", "small"),
    PageBreak(),
    callout([p(verdict)], GREEN if clear_gain else ORANGE, "#ecf8f0" if clear_gain else "#fff4e5"),
    Spacer(1, 4),
    Table([[Image(str(RESULTS / "m2_tree_curve.png"), width=W / 2 - 3 * mm, height=(W / 2 - 3 * mm) * 0.55),
            Image(str(RESULTS / "m2_calibration.png"), width=W / 2 - 13 * mm, height=(W / 2 - 13 * mm) * 0.85)]],
          colWidths=[W / 2, W / 2], style=[("VALIGN", (0, 0), (-1, -1), "MIDDLE")]),
    Image(str(RESULTS / "m2_importance.png"), width=W * 0.8, height=W * 0.8 * 0.46),
    p("Key findings", "h"),
    *bullets([
        f"<b>M1 vs M2 (validation ROC-AUC):</b> version A {m1a.roc_auc:.4f} → {A_val['roc_auc']:.4f} ({gain_a:+.4f}); "
        f"version B {m1b.roc_auc:.4f} → {B_val['roc_auc']:.4f} ({gain_b:+.4f}). Brier for B: {m1b.brier:.4f} → {B_val['brier']:.4f}.",
        f"<b>Approval view (B):</b> approving the safest 50% gives a bad rate of {pct(curve.loc[0.5, 'M2-B'])} with M2 vs "
        f"{pct(curve.loc[0.5, 'M1-B'])} with M1 and {pct(curve.loc[0.5, 'M0'])} with M0.",
        f"<b>More trees stop helping early:</b> B goes from {tc.loc[25, 'B']:.4f} (25 trees) to {tc.loc[100, 'B']:.4f} (100) "
        f"and {tc.loc[500, 'B']:.4f} (500).",
        "<b>What drives the forest (permutation, B):</b> " + ", ".join(
            f"{r.feature} ({r.permutation_auc_drop:+.4f})" for r in perm_top.itertuples()) +
        ". These are the same kinds of drivers M1 found (term, FICO, income, debt), which is a good consistency check.",
        f"<b>Overfitting shows on train:</b> M2-B scores ROC-AUC {B_tr['roc_auc']:.3f} on its own training loans but "
        f"{B_val['roc_auc']:.3f} on validation (M1-B: {lb('M1', 'B', 'train').roc_auc:.3f} vs {m1b.roc_auc:.3f}). Deep trees "
        f"(average depth {info['B']['avg_depth']:.0f}) memorise training loans; only the validation number counts.",
        f"<b>Squeezed probabilities:</b> M2-B over-predicts the safest decile ({calib_b.iloc[0].predicted:.1%} predicted vs "
        f"{calib_b.iloc[0].actual:.1%} actual) and under-predicts the riskiest ({calib_b.iloc[-1].predicted:.1%} vs "
        f"{calib_b.iloc[-1].actual:.1%}). Averaging trees pulls PDs toward the middle, which is why its Brier score is worse than M1's "
        "even though its ranking is slightly better.",
        f"<b>Same ranking, different model:</b> M1-B and M2-B order applicants almost identically (Spearman {spearman:.3f}).",
        f"<b>LendingClub's grades:</b> A beats B by {A_val['roc_auc'] - B_val['roc_auc']:.4f} ROC-AUC, similar to M1.",
        f"<b>Segments (B):</b> ROC-AUC {seg[('B', 36)].roc_auc:.3f} on 36-month vs {seg[('B', 60)].roc_auc:.3f} on 60-month loans; "
        f"mean PD {pct(B_val['mean_pd'])} vs actual {pct(B_val['default_rate'])}, the same 2015 drift as before.",
    ]),
    Spacer(1, 3),
    p("<b>Importance measures.</b> MDI = how much a feature reduced impurity across all splits (fast but favours continuous "
      "features). Permutation = how much validation ROC-AUC drops when that feature is shuffled (20,000-row sample, 3 repeats); "
      "this is the more honest measure.", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/M2_random_forest · see M2_random_forest.ipynb for the executed code")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=14 * mm,
                  bottomMargin=16 * mm, title="M2 - Random Forest").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
