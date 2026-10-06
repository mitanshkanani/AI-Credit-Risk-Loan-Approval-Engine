"""Build models/final/03_decision_layer/F3_decision_layer.pdf from the saved Stage 3 results.

    python models/final/03_decision_layer/build_f3_pdf.py
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
OUT = HERE / "F3_decision_layer.pdf"

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
    "mono": ParagraphStyle("mono", fontName="Courier", fontSize=7.4, leading=9.6, textColor=INK),
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


policy = json.loads((RESULTS / "decision_policy.json").read_text(encoding="utf-8"))
loss = pd.read_csv(RESULTS / "f3_loss_model.csv", dtype={"term": str}).set_index("term")
bands = pd.read_csv(RESULTS / "f3_risk_bands.csv").set_index("band")
decisions = pd.read_csv(RESULTS / "f3_decisions.csv").set_index("decision")
backtest = pd.read_csv(RESULTS / "f3_el_backtest.csv", index_col=0)
impact = pd.read_csv(RESULTS / "f3_business_impact.csv").set_index("policy")
checks = pd.read_csv(RESULTS / "f3_checks.csv")
examples = json.loads((RESULTS / "f3_examples.json").read_text(encoding="utf-8"))
t = policy["thresholds"]
lm = policy["loss_model"]
tg = policy["business_targets"]
ap, rv, dc = decisions.loc["APPROVE"], decisions.loc["REVIEW"], decisions.loc["DECLINE"]
base, strict = impact.iloc[0], impact.iloc[1]
good_turned_away = int(base.approved_loans - strict.approved_loans - strict.defaults_avoided)
all_pass = bool((checks.loc[checks["required"], "result"] == "PASS").all())

headline = (
    f"<b>Result:</b> <b>APPROVE</b> if PD ≤ {pct(t['approve_max_pd'])} ({pct(ap.share)} of applicants, {pct(ap.default_rate)} default), "
    f"<b>REVIEW</b> in between ({pct(rv.share)}), <b>DECLINE</b> if PD ≥ {pct(t['decline_min_pd'])} ({pct(dc.share)}, "
    f"{pct(dc.default_rate)} default). Predicted expected loss is <b>{backtest.loc['all', 'ratio']:.2f}×</b> the real loss. "
    + ("All required checks pass." if all_pass else "NOT all required checks pass.")
)

loss_rows = [["Term", "Defaulted training loans", "Share still owed at default (EAD share)", "Loss given default (LGD)",
              "Loss per $ funded (EAD share × LGD)"]]
for term in ("36", "60", "all"):
    r = loss.loc[term]
    loss_rows.append([f"{term} months" if term != "all" else "all", f"{int(r.defaulted_loans):,}", pct(r.ead_share), pct(r.lgd),
                      pct(r.loss_share_of_funded)])

band_rows = [["Band", "PD range", "Share", "Mean PD", "Actual default", "Loss per $ funded"]]
for b in policy["risk_bands"]:
    r = bands.loc[b["band"]]
    rng = f"≥ {b['pd_lower']:.0%}" if b["pd_upper"] >= 1 else f"{b['pd_lower']:.0%} – {b['pd_upper']:.0%}"
    band_rows.append([f"{b['band']} {b['label']}", rng, pct(r.share), pct(r.mean_pd), pct(r.default_rate), pct(r.loss_rate)])

rules = {"APPROVE": f"PD ≤ {pct(t['approve_max_pd'])}", "REVIEW": "in between", "DECLINE": f"PD ≥ {pct(t['decline_min_pd'])}"}
dec_rows = [["Decision", "Rule", "Share", "Mean PD", "Actual default", "Expected loss", "Real loss"]]
for d, r in decisions.iterrows():
    dec_rows.append([d, rules[d], pct(r.share), pct(r.mean_pd), pct(r.default_rate), money(r.expected_loss_m), money(r.realised_loss_m)])

bt_rows = [["Group", "Expected", "Real", "Ratio"]]
for g, r in backtest.loc[["all", "term 36", "term 60"]].iterrows():
    bt_rows.append([g + (" months" if g.startswith("term") else ""), money(r.expected_loss_m), money(r.realised_loss_m), f"{r.ratio:.3f}"])

imp_rows = [["Policy on the 162,745 validation loans", "Approved", "Default rate", "Real loss", "Loss per $ funded", "Loss avoided"]]
for name, r in impact.iterrows():
    imp_rows.append([name, pct(r.approval_rate), pct(r.default_rate), money(r.realised_loss_m), pct(r.loss_rate),
                     money(r.loss_avoided_m) if r.loss_avoided_m else "–"])

ex = examples[-1]


def q(items):
    return ", ".join(f'"{s}"' for s in items)


ex_json = "<br/>".join([
    "{",
    f'&nbsp;"probability_of_default": {ex["probability_of_default"]},',
    f'&nbsp;"risk_band": "{ex["risk_band"]}",',
    f'&nbsp;"decision": "{ex["decision"]}",',
    f'&nbsp;"loan_amount": {ex["loan_amount"]:.0f}, "term_months": {ex["term_months"]},',
    f'&nbsp;"expected_loss_usd": {ex["expected_loss_usd"]:.2f},',
    '&nbsp;"reasons": [',
    *[f'&nbsp;&nbsp;"{r}"' + ("," if i < len(ex["reasons"]) - 1 else "],") for i, r in enumerate(ex["reasons"])],
    f'&nbsp;"strengths": [{q(ex["strengths"])}]',
    "}",
])

story = [
    p("Final model · Stage 3: the decision layer", "title"),
    p("From a probability to an action · validation split only (2015 H1) · run locally · test still sealed", "sub"),
    callout([p(headline)], GREEN, "#ecf8f0"),
    p("A. How much is lost when a loan defaults?", "h"),
    p(f"Measured on the <b>{lm['defaulted_loans']:,} defaulted training loans</b> (2012-08..2014-12), using repayment and recovery columns "
      "from the raw file. These are outcomes known only after a loan ends, so they were never model inputs. <b>EAD</b> (exposure at "
      "default) = principal still owed when the borrower stops paying; <b>LGD</b> (loss given default) = the part of it never recovered, "
      "after collection-agency fees."),
    table(loss_rows, [22 * mm, 30 * mm, 46 * mm, 36 * mm, W - 134 * mm], BLUE),
    p(f"<b>Expected loss (EL) = PD × loan amount × EAD share × LGD.</b> Example: a $10,000 loan with PD 20% → "
      f"0.20 × 10,000 × {lm['ead_share']['36']:.2f} × {lm['lgd']['36']:.2f} = <b>${0.2 * 10000 * lm['ead_share']['36'] * lm['lgd']['36']:,.0f}</b> "
      f"for 36 months, ${0.2 * 10000 * lm['ead_share']['60'] * lm['lgd']['60']:,.0f} for 60 months (more principal is still owed when "
      "a longer loan defaults). Recoveries are small: a quarter of defaults recovered nothing."),
    p("B. Risk bands", "h"),
    p("Seven fixed PD ranges give every applicant a short label, like a grade. The requirement (fixed beforehand) is that the actual "
      "default rate rises from band to band. It does, and each band's mean PD matches its actual default rate."),
    table(band_rows, [36 * mm, 24 * mm, 22 * mm, 24 * mm, 28 * mm, W - 134 * mm], GREY),
    Image(str(RESULTS / "f3_risk_bands.png"), width=W * 0.8, height=W * 0.8 * 3.8 / 8),
    PageBreak(),
    p("C. Approve / review / decline", "h"),
    p("The thresholds come from two business targets written in the plan before any result was seen. They were searched on a 0.5-point "
      f"PD grid using out-of-fold calibrated PDs: <b>APPROVE</b> the largest group whose actual default rate is ≤ "
      f"{tg['approved_book_max_default_rate']:.0%}; <b>DECLINE</b> the riskiest group whose default rate is ≥ "
      f"{tg['decline_group_min_default_rate']:.0%} (two in five default); send the rest to a person for <b>REVIEW</b>."),
    table(dec_rows, [22 * mm, 26 * mm, 18 * mm, 20 * mm, 25 * mm, 28 * mm, W - 139 * mm], BLUE),
    Image(str(RESULTS / "f3_thresholds.png"), width=W * 0.9, height=W * 0.9 * 3.9 / 11),
    Table([[[p("D. Does expected loss match reality?", "h"),
             table(bt_rows, [24 * mm, 18 * mm, 18 * mm, W * 0.41 - 60 * mm], GREY),
             p("Required: within ±10% overall (by decision group: see table C). Recoveries on 2015 loans were still coming in when the data ends (2018), so the real "
               "loss is, if anything, slightly overstated.", "small")],
            [p("E. What it returns (a declined applicant)", "h"),
             callout([p(ex_json, "mono")], BLUE, "#eef3fc", width=W * 0.56),
             p(f"Actual outcome: {ex['actual_outcome']}. Reasons come from the Stage 2 SHAP reason codes.", "small")]]],
          colWidths=[W * 0.43, W * 0.57], style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
    p("F. Business impact on validation (losses only)", "h"),
    table(imp_rows, [64 * mm, 19 * mm, 21 * mm, 21 * mm, 25 * mm, W - 150 * mm], GREEN),
    p("What we learned", "h"),
    *bullets([
        f"The strict policy cuts real losses from {money(base.realised_loss_m)} to {money(strict.realised_loss_m)} ({pct(strict.loss_rate)} "
        f"instead of {pct(base.loss_rate)} of every dollar lent), but it also turns away {good_turned_away:,} borrowers who repaid. Their "
        "lost interest is not counted here because the model does not set prices; a real lender would weigh both.",
        f"The approve threshold ({pct(t['approve_max_pd'])}) falls inside band R5 (20–30%), so R5 is split across all three decisions. "
        "Bands describe risk; thresholds are business choices. Both live in one editable file, <i>decision_policy.json</i>.",
        f"60-month loans: expected loss is {backtest.loc['term 60', 'ratio']:.2f}× the real loss (36-month: "
        f"{backtest.loc['term 36', 'ratio']:.2f}×). This matches the Stage 1 finding that 60-month PDs run about 1 point low; "
        "Stage 5 (test) will show whether a term-specific correction is needed.",
    ]),
    p("Next: Stage 4, one scoring package (raw application → preprocessing → model → calibration → this decision layer → JSON).", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/final/03_decision_layer · see F3_decision_layer.ipynb and results/decision_policy.json")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="Final model - Stage 3").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
