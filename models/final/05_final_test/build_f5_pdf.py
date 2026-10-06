"""Build models/final/05_final_test/F5_final_test.pdf from the saved Stage 5 results.

    python models/final/05_final_test/build_f5_pdf.py
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "F5_final_test.pdf"

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


def money(m):
    return f"${m:,.1f}M"


summary = json.loads((RESULTS / "f5_summary.json").read_text(encoding="utf-8"))
metrics = pd.read_csv(RESULTS / "f5_metrics.csv", index_col=0)
by_term = pd.read_csv(RESULTS / "f5_by_term.csv").set_index("term")
decisions = pd.read_csv(RESULTS / "f5_decisions.csv").set_index("decision")
impact = pd.read_csv(RESULTS / "f5_business_impact.csv").set_index("policy")
states = pd.read_csv(RESULTS / "f5_states.csv")
stress = pd.read_csv(RESULTS / "f5_stress.csv")
criteria = pd.read_csv(RESULTS / "f5_criteria.csv")
m = summary["metrics"]
lc = summary["lendingclub_int_rate_roc_auc"]
met, total = summary["criteria_met"], summary["criteria_total"]

headline = (
    f"<b>Result on the sealed test set (2015 H2, {summary['test_rows']:,} loans, opened once):</b> ROC-AUC <b>{m['roc_auc']:.4f}</b> "
    f"(validation {metrics.loc['roc_auc'].iloc[0]:.4f}), mean PD {pct(m['mean_pd'])} vs actual {pct(m['default_rate'])}. The engine ranks "
    f"risk better than <b>LendingClub's own interest rate</b> (ROC-AUC {lc:.4f}, +{m['roc_auc'] - lc:.3f}). "
    f"<b>{met} of {total}</b> pre-registered criteria met; the miss is the known 60-month calibration gap."
)

metric_names = {"default_rate": "Default rate", "mean_pd": "Mean predicted PD", "calibration_gap": "Calibration gap (PD − actual)",
                "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "ks": "KS", "brier": "Brier score (lower = better)",
                "log_loss": "Log loss (lower = better)", "ece": "Expected calibration error"}
metric_rows = [["Metric", "Validation (2015 H1)", "TEST (2015 H2)", "Change"]]
for k, name in metric_names.items():
    v, t, c = metrics.loc[k]
    f = pct if k in ("default_rate", "mean_pd", "calibration_gap", "ece") else (lambda x: f"{x:.4f}")
    metric_rows.append([name, f(v), f"<b>{f(t)}</b>", f"{c * 100:+.2f} pts" if f is pct else f"{c:+.4f}"])

crit_rows = [["Pre-registered criterion (committed before opening the test)", "Test value", "Result"]]
for r in criteria.itertuples():
    color = "#2e9b5b" if r.result == "PASS" else "#d4840f"
    value = r.value if len(str(r.value)) < 40 else "3.0% → 7.4% → … → 52.9%"
    crit_rows.append([r.criterion, value, f'<font color="{color}"><b>{r.result}</b></font>'])

term_rows = [["Term", "Loans", "Mean PD", "Actual default", "Gap", "ROC-AUC"]]
for t, r in by_term.iterrows():
    term_rows.append([f"{t} months", f"{int(r.loans):,}", pct(r.mean_pd), pct(r.default_rate), f"{r.gap_pts:+.2f} pts", f"{r.roc_auc:.4f}"])

dec_rows = [["Decision", "Share (validation)", "Share (test)", "Default (validation)", "Default (test)", "Expected ÷ real loss"]]
for d, r in decisions.iterrows():
    dec_rows.append([d, pct(r.validation_share), pct(r.share), pct(r.validation_default_rate), pct(r.default_rate), f"{r.el_ratio:.2f}"])

imp_rows = [["Policy on the test loans", "Approved", "Default rate", "Real loss", "Loss per $ lent", "Loss avoided"]]
for name, r in impact.iterrows():
    imp_rows.append([name, pct(r.approval_rate), pct(r.default_rate), money(r.realised_loss_m), pct(r.loss_rate),
                     money(r.loss_avoided_m) if r.loss_avoided_m else "–"])

stress_rows = [["Year (stress)", "Loans", "ROC-AUC"]] + [[str(r.year), f"{r.loans:,}", f"{r.roc_auc:.4f}"] for r in stress.itertuples()]
flagged = states.loc[states["flag"], "state"].tolist()

story = [
    p("Final model · Stage 5: the final test", "title"),
    p("Sealed 2015 H2 test split opened once · frozen engine v1.0.0 · criteria committed beforehand (commit 8c5b9ab)", "sub"),
    callout([p(headline)], GREEN, "#ecf8f0"),
    p("A. How the test was run", "h"),
    p("The test loans were never used for any choice: not for training, early stopping, calibration, thresholds or the zip-code "
      "decision. They were scored with the frozen bundle (checksums verified on load). As a final end-to-end proof, 5,000 test loans "
      "were also scored by the deployable engine from their <b>original raw CSV rows</b>: maximum PD difference "
      f"<b>{summary['end_to_end_max_pd_diff']:.1f}</b>."),
    p("B. Test vs validation", "h"),
    table(metric_rows, [62 * mm, 38 * mm, 38 * mm, W - 138 * mm], BLUE),
    p("Test scores are slightly <i>better</i> than validation. That is not leakage: the test months simply ranked more easily. "
      "LendingClub's own interest rate improves by the same amount (ROC-AUC 0.7037 on validation → 0.7148 on test, +0.011, vs our "
      "+0.011). The engine kept its lead over LendingClub in both periods.", "small"),
    p("C. Pre-registered criteria", "h"),
    table(crit_rows, [92 * mm, 52 * mm, W - 144 * mm], GREEN),
    p("D. Calibration by loan term: the one miss", "h"),
    table(term_rows, [24 * mm, 24 * mm, 26 * mm, 30 * mm, 24 * mm, W - 128 * mm], GREY),
    p(f"60-month loans default <b>{abs(by_term.loc[60, 'gap_pts']):.1f} points</b> more often than predicted (validation: about 1 point). "
      "The ranking within 60-month loans is fine; only the level is low. The rule says nothing changes after the test, so v1.0.0 "
      "ships as is and the gap is <b>documented</b>. The remedy for a v1.1 would be a term-specific calibrator fitted on newer data.", "b"),
    PageBreak(),
    Image(str(RESULTS / "f5_test_calibration.png"), width=W, height=W * 3.8 / 11),
    p("E. Decisions and losses on test", "h"),
    table(dec_rows, [24 * mm, 29 * mm, 24 * mm, 31 * mm, 26 * mm, W - 134 * mm], BLUE),
    table(imp_rows, [60 * mm, 20 * mm, 22 * mm, 22 * mm, 24 * mm, W - 148 * mm], GREEN),
    p(f"The approve threshold chosen on validation behaves the same on test: {pct(decisions.loc['APPROVE', 'share'])} approved with "
      f"{pct(decisions.loc['APPROVE', 'default_rate'])} default (target ≤ 12%). Expected loss is "
      f"{summary['expected_over_realised_loss']:.2f}× the real loss, close for approved loans and lower for risky ones, the same "
      "60-month under-prediction showing through.", "b"),
    Table([[[p("F. States", "h"),
             p(f"{len(states)} states with ≥ 1,000 test loans: calibration gaps {states['gap_pts'].min():+.1f} to "
               f"{states['gap_pts'].max():+.1f} points. Beyond ±2: {', '.join(flagged)}. Of Stage 2's flags, LA and OR are flagged again "
               "and TN is fine. Small states move more between periods, so this supports monitoring, not a fix.", "b")],
            [p("G. Stress: 2016–2018", "h"),
             table(stress_rows, [26 * mm, 22 * mm, 20 * mm], GREY),
             p("Ranking only (labels are biased toward early outcomes). It holds up 1–3 years past training.", "small")]]],
          colWidths=[W * 0.56, W * 0.44], style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
    p("What we learned", "h"),
    *bullets([
        f"The engine generalises to unseen months: ROC-AUC {m['roc_auc']:.3f}, overall calibration within {abs(m['calibration_gap']) * 100:.1f} "
        "points, and decisions behave as designed.",
        "It beats LendingClub's own risk pricing at ranking these loans, without ever seeing the grade or interest rate.",
        "Known limitation, now confirmed on test: 60-month loans are under-predicted by about 2.5 points. It is documented, not fixed, "
        "because changing the model after seeing the test would make this test meaningless.",
    ]),
    p("Next: Stage 6, the live demo (FastAPI + web page, one public link).", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/final/05_final_test · see F5_final_test.ipynb · test split opened once")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="Final model - Stage 5").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
