"""Build models/final/04_scoring_package/F4_scoring_package.pdf from the saved Stage 4 results.

    python models/final/04_scoring_package/build_f4_pdf.py
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
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE / "F4_scoring_package.pdf"

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
    "c": ParagraphStyle("c", fontName="Body", fontSize=8.2, leading=10.6, textColor=INK),
    "cw": ParagraphStyle("cw", fontName="Body-Bold", fontSize=8.4, leading=10.8, textColor=colors.white),
    "small": ParagraphStyle("sm", fontName="Body", fontSize=7.8, leading=10.2, textColor=MUTED),
    "mono": ParagraphStyle("mono", fontName="Courier", fontSize=7.3, leading=9.4, textColor=INK),
}
W = A4[0] - 34 * mm


def p(text, style="b"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["li"], bulletText="•") for t in items]


def table(rows, widths, color=GREY):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cw" if r == 0 else "c"]) for c in row] for r, row in enumerate(rows)]
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


manifest = json.loads((HERE / "bundle" / "manifest.json").read_text(encoding="utf-8"))
report = json.loads((RESULTS / "f4_engine_test.json").read_text(encoding="utf-8"))
example = json.loads((RESULTS / "f4_example_output.json").read_text(encoding="utf-8"))
real = json.loads((RESULTS / "f4_real_examples.json").read_text(encoding="utf-8"))
latency = pd.read_csv(RESULTS / "f4_latency.csv")
contract = pd.read_csv(RESULTS / "f4_input_contract.csv")
checks = report["checks"]
required = [c for c in checks if c["required"]]
total_ms = latency.iloc[-1]["ms"]
sizes = {f: (HERE / "bundle" / f).stat().st_size / 1e6 for f in manifest["sha256"]}

headline = (
    f"<b>Result:</b> one engine, <b>CreditRiskEngine</b>, takes a raw application and returns the decision JSON. "
    f"<b>{sum(c['status'] == 'PASS' for c in required)} of {len(required)} required checks pass</b>. The original raw CSV rows of 2,000 "
    "validation loans reproduce Stage 3's PD, band, decision and expected loss <b>exactly</b> (difference 0.0). "
    f"Scoring one application takes about {total_ms / 1000:.1f} s."
)

pipeline_rows = [["Step", "File in bundle/", "From", "Size"],
                 ["1. Feature engineering, imputation, encoding", "preprocessor_v2.joblib", "Preprocessing V2 (fitted on train)",
                  f"{sizes['preprocessor_v2.joblib']:.1f} MB"],
                 ["2. Probability of default (146 features, 1,931 trees)", "final_model.txt", "Stage 2: M3-B-mono-nozip",
                  f"{sizes['final_model.txt']:.1f} MB"],
                 ["3. Calibration (Platt: 2 numbers)", "calibrator.json", "Stage 2", "&lt; 1 KB"],
                 ["4. Risk band, decision, expected loss", "decision_policy.json", "Stage 3", "2 KB"],
                 ["5. Reason codes (SHAP → sentences)", "code: reason_codes.py", "Stage 2", "–"],
                 ["Integrity", "manifest.json", "checksums, versions, features", "–"]]

short = {
    "engine loads in a fresh process; all bundle checksums match": "Loads in a fresh process; checksums match",
    "a tampered bundle file is refused": "A tampered bundle file is refused",
    "single-application scoring agrees with batch scoring (20 loans)": "Single scoring = batch scoring (20 loans)",
    "sparse hand-written application -> complete JSON, no NaN": "Sparse hand-written application → complete JSON",
    "impossible inputs are rejected with a message": "Impossible inputs rejected with a message (6 cases)",
    "same input -> same output": "Same input → same output",
    "higher FICO never raises PD; higher DTI never lowers it": "Higher FICO never raises PD; higher DTI never lowers it",
}
detail = {
    0: "4 files verified in 0.2 s; scikit-learn 1.9.0",
    1: "calibrator edited → engine refuses to start",
    2: "max PD difference 0.0; bands, decisions, EL identical",
    3: "max difference 0.00005 (JSON rounds PD to 4 decimals)",
    4: f"PD {report['example_output']['probability_of_default']}, {report['example_output']['decision']}",
    5: "e.g. term 48 → \"term must be 36 or 60 (months)\"",
    6: "",
    7: f"FICO 640→850: PD {report['fico_sweep'][0]:.3f}→{report['fico_sweep'][-1]:.3f}; DTI 0→40: {report['dti_sweep'][0]:.3f}→{report['dti_sweep'][-1]:.3f}",
    8: f"median {report['latency_ms_median']:.0f} ms (warning only)",
}
check_rows = [["Check (run as its own process: tests/engine_fresh_session_test.py)", "Result", "Detail"]]
for i, c in enumerate(checks):
    name = short.get(c["check"], c["check"].replace("raw CSV rows of 2,000 validation loans reproduce Stage 3 exactly",
                                                    "<b>Raw CSV rows of 2,000 validation loans reproduce Stage 3</b>"))
    color = "#2e9b5b" if c["status"] == "PASS" else "#d4840f"
    check_rows.append([name, f'<font color="{color}"><b>{c["status"]}</b></font>', detail.get(i, "")])

contract_rows = [["Input group", "Fields", "Examples"]]
for r in contract.itertuples():
    contract_rows.append([r.group, str(r.fields), r.examples])

ex_lines = ["{",
            f'&nbsp;"probability_of_default": {example["probability_of_default"]},',
            f'&nbsp;"risk_band": "{example["risk_band"]}", "risk_band_label": "{example["risk_band_label"]}",',
            f'&nbsp;"decision": "{example["decision"]}", "decision_rule": "{example["decision_rule"]}",',
            f'&nbsp;"loan_amount": {example["loan_amount"]:.0f}, "term_months": {example["term_months"]},',
            f'&nbsp;"expected_loss_usd": {example["expected_loss_usd"]}, "expected_loss_rate": {example["expected_loss_rate"]},',
            '&nbsp;"reasons": [']
ex_lines += [f'&nbsp;&nbsp;{{"reason": "{r["reason"]}", "impact": {r["impact"]}}}' + ("," if i < len(example["reasons"]) - 1 else "],")
             for i, r in enumerate(example["reasons"])]
ex_lines += ['&nbsp;"strengths": [...2 items...],',
             f'&nbsp;"missing_fields": [...{len(example["missing_fields"])} bureau fields...],',
             '&nbsp;"warnings": [...], "model": {"name": "M3-B-mono-nozip", "version": "1.0.0"}',
             "}"]


def real_block(e):
    head = (f"<b>{e['decision']}</b> · PD {pct(e['probability_of_default'])} · {e['risk_band']}<br/>"
            f"${e['loan_amount']:,.0f}, {e['term_months']} months · EL ${e['expected_loss_usd']:,.0f}<br/>"
            f"<i>actually {e['actual_outcome']}</i>")
    return [p(head, "c")] + [p(f"▲ {r}", "c") for r in e["reasons"][:3]] + [p(f"▼ {r}", "c") for r in e["strengths"][:1]]


lat_rows = [["Step", "ms"]] + [[r.step, f"{r.ms:.0f}" if r.ms >= 1 else f"{r.ms:.1f}"] for r in latency.itertuples()]

story = [
    p("Final model · Stage 4: the scoring package", "title"),
    p("Raw application in → decision JSON out · Python 3.14 + scikit-learn 1.9.0 · validation data only · test still sealed", "sub"),
    callout([p(headline)], GREEN, "#ecf8f0"),
    p("A. What is in the package", "h"),
    p("Everything the engine needs sits in one folder, <i>models/final/04_scoring_package/bundle/</i> (about 10 MB, now committed so the "
      "live demo can be deployed straight from GitHub). <i>manifest.json</i> stores each file's SHA-256 checksum and the library versions. "
      "The engine <b>refuses to start</b> if a file was changed or if scikit-learn is not 1.9.0, the version that fitted the preprocessor "
      "(the Stage 1–3 notebooks only read already-transformed data, so their scikit-learn version did not matter; the engine transforms raw input, so it does)."),
    table(pipeline_rows, [64 * mm, 38 * mm, 50 * mm, W - 152 * mm], BLUE),
    p("B. What an application must contain", "h"),
    p(f"Only <b>loan amount</b> and <b>term</b> (36 or 60) are required. The model reads {len(manifest['model_input_fields'])} raw fields; "
      "any not supplied get the training median or the \"missing\" encoding and are listed back in <i>missing_fields</i>, so the caller "
      f"always knows what was assumed. {len(manifest['ignored_raw_fields'])} raw LendingClub columns are ignored: LendingClub's own grade, "
      "sub-grade, interest rate and instalment (version B), zip code (dropped in Stage 2), and joint-application fields. "
      "Impossible values (term 48, FICO 950, negative income, a malformed date) are rejected with a readable message."),
    table(contract_rows, [42 * mm, 14 * mm, W - 56 * mm], GREY),
    p("C. Fresh-session test", "h"),
    table(check_rows, [76 * mm, 21 * mm, W - 97 * mm], GREEN),
    PageBreak(),
    p("D. What the engine returns", "h"),
    p("A hand-written applicant (60 months, DTI 27.5%, FICO 665, revolving use 88%, 3 recent inquiries) who left 44 bureau fields blank:"),
    callout([p("<br/>".join(ex_lines), "mono")], BLUE, "#eef3fc"),
    p("Three real validation applicants, scored from their original raw CSV rows (▲ raises risk, ▼ lowers it)", "h"),
    Table([[real_block(e) for e in real]], colWidths=[W / 3] * 3,
          style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
                 ("LINEBEFORE", (1, 0), (2, 0), 0.5, colors.HexColor("#d1d5db")), ("LEFTPADDING", (0, 0), (-1, -1), 5)]),
    Table([[[p("E. Speed (one application)", "h"), table(lat_rows, [62 * mm, 16 * mm], GREY)],
            [p("Why 0.4 s?", "h"),
             p("scikit-learn's target encoder rebuilds its lookup table of 143,539 job titles and 31,812 loan titles on <b>every</b> "
               "call. That is fine for a web demo. If speed ever matters, the tables can be cached as a dictionary at start-up; we keep "
               "the exact scikit-learn path because it is proven to reproduce Stage 3 to the last digit. The model itself (1 ms) and its "
               "SHAP explanation (11 ms) are fast.", "b")]]],
          colWidths=[W * 0.48, W * 0.52], style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]),
    p("What we learned", "h"),
    *bullets([
        "The engine is the same code path that produced every validation number in Stages 1–3: proven by re-scoring 2,000 loans from "
        "the untouched raw file with zero difference.",
        "Blank bureau fields are allowed but matter: the hand-written applicant's PD rests partly on assumed values. For the live demo "
        "(Stage 6), the form will start from complete real profiles, and the response will show <i>missing_fields</i>.",
        "Versions are pinned in <i>requirements-engine.txt</i>, so the Docker image for Stage 6 rebuilds exactly this environment.",
    ]),
    p("Next: Stage 5, open the sealed 2015 H2 test set once and score it with this frozen bundle.", "small"),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Body", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(17 * mm, 10 * mm, "models/final/04_scoring_package · src/engine/scorer.py · tests/engine_fresh_session_test.py")
    canvas.drawRightString(A4[0] - 17 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=17 * mm, rightMargin=17 * mm, topMargin=13 * mm,
                  bottomMargin=16 * mm, title="Final model - Stage 4").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
