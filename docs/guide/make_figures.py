"""Figures for the project guide (architecture, preprocessing flow, deployment, ROC-AUC ladder, SHAP waterfall).

    python docs/guide/make_figures.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

INK, MUTED = "#1f2937", "#6b7280"
C = {"data": "#e8eefc", "prep": "#e7f5ec", "model": "#fdf3e2", "dec": "#fbeae7", "dep": "#efe9fb", "grey": "#f3f4f6"}
EDGE = {"data": "#3b6fd4", "prep": "#2e9b5b", "model": "#d4840f", "dec": "#c2412d", "dep": "#7c4dcc", "grey": "#9ca3af"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def box(ax, x, y, w, h, title, body="", kind="grey", fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=C[kind], ec=EDGE[kind], lw=1.4))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.05, body, ha="center", va="top", fontsize=fs - 1.6, color=INK, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, text="", color="#4b5563"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.3, color=color))
    if text:
        ax.text((x1 + x2) / 2 + 0.008, (y1 + y2) / 2, text, fontsize=7.6, color=MUTED, va="center")


def canvas(w=8.3, h=11.0):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ------------------------------------------------------------------ 1. end-to-end architecture
fig, ax = canvas(8.3, 10.6)
ax.text(0.5, 0.985, "End-to-end architecture: from raw CSV to a live decision", ha="center", va="top", fontsize=15, fontweight="bold")
ax.text(0.5, 0.958, "left = built once, offline (training)        right = runs for every web request (serving)", ha="center",
        va="top", fontsize=9.5, color=MUTED)
# offline column
X0, W0 = 0.03, 0.44
box(ax, X0, 0.84, W0, 0.095, "1. Raw data", "LendingClub accepted loans 2007-2018\n2,260,701 loans × 151 columns", "data")
box(ax, X0, 0.715, W0, 0.105, "2. Preprocessing V1 (Steps 1-10)", "target + eligible loans (1.35M) · leakage audit\n"
    "drop 49 future / id / admin columns → 103\nquality checks · missing-value plan", "prep")
box(ax, X0, 0.565, W0, 0.13, "3. Preprocessing V2 (fitted pipeline)", "restore id / issue_d / target (0 mismatches)\n"
    "temporal split: train 2012-08..14 · val 15H1 · test 15H2\nfeature engineering · TRAIN-only fit:\nfilter · median+flags · one-hot · target enc.\n"
    "→ 190 features · 72 audits pass", "prep")
box(ax, X0, 0.42, W0, 0.125, "4. Model ladder (Kaggle CPU)", "M0 dummy 0.500 · M1 logistic 0.725\nM2 random forest 0.727 · M3 LightGBM 0.736\n"
    "M4 XGBoost 0.736 (validation ROC-AUC, version B)\n→ LightGBM chosen", "model")
box(ax, X0, 0.265, W0, 0.135, "5. Finalisation (Stages 1-3)", "monotonic constraints (0 violations)\nPlatt calibration (mean PD = actual)\n"
    "zip code dropped (fairness) → 146 features\nSHAP reason codes · decision policy\nrisk bands · EL = PD × EAD × LGD", "dec")
box(ax, X0, 0.135, W0, 0.11, "6. Scoring bundle (Stage 4) + test (Stage 5)", "preprocessor + model + calibrator + policy\n"
    "+ manifest.json (SHA-256, versions)\nsealed test: ROC-AUC 0.746 (opened once)", "dec")
for y1, y2 in [(0.84, 0.82), (0.715, 0.695), (0.565, 0.545), (0.42, 0.40), (0.265, 0.245)]:
    arrow(ax, X0 + W0 / 2, y1, X0 + W0 / 2, y2)
# online column
X1, W1 = 0.53, 0.44
box(ax, X1, 0.84, W1, 0.095, "A. Browser / API client", "web form (index.html) or any HTTP client\nPOST /api/score  {loan_amnt, term, fico, ...}", "dep")
box(ax, X1, 0.715, W1, 0.105, "B. Render (free web service)", "Docker container built from GitHub (render.yaml)\n"
    "python:3.14-slim + pinned libraries\nuvicorn → FastAPI app (deploy/app/main.py)", "dep")
box(ax, X1, 0.565, W1, 0.13, "C. Guardrails (v1.3, input side)", "validate types, categories, required fields\n"
    "unknown job/loan title → 'not provided'\nfill loan title from purpose\nscope checks: FICO ≥ 660, DTI ≤ 40, loan ≤ 50% income...", "dep")
box(ax, X1, 0.42, W1, 0.125, "D. CreditRiskEngine.score()", "preprocessor.transform → 146 features\nLightGBM → raw PD → Platt → calibrated PD\n"
    "decision policy → band, decision, expected loss\nSHAP pred_contrib → reasons / strengths", "model")
box(ax, X1, 0.265, W1, 0.135, "E. Guardrails (output side)", "out of scope → REFER, score hidden\n"
    "< 50% bureau data → APPROVE becomes REVIEW\nderogatory / missing never shown as strength", "dec")
box(ax, X1, 0.135, W1, 0.11, "F. JSON response → result card", "probability_of_default · risk_band · decision\n"
    "expected_loss_usd · reasons · strengths\nmodel_decision · guardrail_notes · latency", "dep")
for y1, y2 in [(0.84, 0.82), (0.715, 0.695), (0.565, 0.545), (0.42, 0.40), (0.265, 0.245)]:
    arrow(ax, X1 + W1 / 2, y1, X1 + W1 / 2, y2)
arrow(ax, X0 + W0 + 0.005, 0.19, X1 - 0.005, 0.48, "", "#7c4dcc")
ax.text(0.488, 0.255, "bundle\nloaded\nonce at\nstart-up", fontsize=7.4, color="#7c4dcc", ha="center", va="center")
ax.text(0.5, 0.075, "One GitHub repository holds both sides. A push to main makes Render rebuild the Docker image and redeploy.\n"
        "Training data never goes into the container: only the ~10 MB bundle (checksummed) and the app code.",
        ha="center", va="top", fontsize=9.2, color=INK)
fig.savefig(OUT / "architecture.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 2. preprocessing flow
fig, ax = canvas(8.3, 7.4)
ax.text(0.5, 0.98, "Preprocessing pipeline (V1 audit steps → V2 fitted pipeline)", ha="center", va="top", fontsize=14, fontweight="bold")
steps = [
    ("Raw CSV", "2.26M × 151", "data"), ("Target + eligible", "1.35M resolved loans", "prep"),
    ("Column audit", "type, missing, unique", "prep"), ("Leakage review", "94 app-time / 42 future", "prep"),
    ("Remove 49 columns", "→ 103 candidates", "prep"), ("Quality checks", "nothing to drop", "prep"),
    ("Representation plan", "per-column treatment", "prep"), ("Missing-value plan", "learn from TRAIN only", "prep"),
    ("Restore keys", "id, issue_d, target", "model"), ("Temporal split", "train/val/test/stress", "model"),
    ("Feature engineering", "credit_history_months...", "model"), ("Fit on TRAIN only", "filter·impute·encode", "model"),
    ("Export", "190 features, Parquet", "dec"), ("72 audits", "0 fail · 0 warn", "dec"), ("Fresh-session test", "raw → X exact", "dec"),
]
cols, w, h = 3, 0.29, 0.085
for i, (t, b, k) in enumerate(steps):
    r, c = divmod(i, cols)
    c = c if r % 2 == 0 else cols - 1 - c          # snake layout
    x, y = 0.04 + c * 0.32, 0.84 - r * 0.165
    box(ax, x, y, w, h, f"{i + 1}. {t}", b, k, fs=10)
    if i + 1 < len(steps):
        r2, c2 = divmod(i + 1, cols)
        c2 = c2 if r2 % 2 == 0 else cols - 1 - c2
        x2, y2 = 0.04 + c2 * 0.32, 0.84 - r2 * 0.165
        if r2 == r:
            arrow(ax, x + (w if c2 > c else 0), y + h / 2, x2 + (0 if c2 > c else w), y2 + h / 2)
        else:
            arrow(ax, x + w / 2, y, x2 + w / 2, y2 + h)
ax.text(0.5, 0.03, "blue = data · green = V1 notebook (Steps 1-10, audits) · orange = V2 build · red = outputs and checks",
        ha="center", fontsize=9, color=MUTED)
fig.savefig(OUT / "preprocessing_flow.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 3. deployment
fig, ax = canvas(8.3, 6.2)
ax.text(0.5, 0.98, "Deployment: GitHub → Render → Docker → FastAPI", ha="center", va="top", fontsize=14, fontweight="bold")
box(ax, 0.03, 0.66, 0.27, 0.2, "Laptop", "code + tests\nbuild_bundle.py\ngit push origin main", "grey")
box(ax, 0.365, 0.66, 0.27, 0.2, "GitHub repo", "Dockerfile · render.yaml\nsrc/ · deploy/app/\nbundle/ (~10 MB)", "data")
box(ax, 0.70, 0.66, 0.27, 0.2, "Render build", "docker build:\npython:3.14-slim + libgomp1\npip install pinned versions\nCOPY only engine + bundle + app", "dep")
arrow(ax, 0.30, 0.76, 0.365, 0.76, "")
arrow(ax, 0.635, 0.76, 0.70, 0.76, "")
ax.text(0.33, 0.79, "push", fontsize=8, color=MUTED, ha="center")
ax.text(0.667, 0.79, "auto-deploy", fontsize=8, color=MUTED, ha="center")
box(ax, 0.20, 0.13, 0.6, 0.42, "Running container (free instance, ~240 MB RAM)", "", "dep", fs=11)
box(ax, 0.24, 0.36, 0.24, 0.13, "uvicorn :$PORT", "ASGI web server", "grey", fs=9.5)
box(ax, 0.52, 0.36, 0.24, 0.13, "FastAPI app", "/ · /docs · /api/health\n/api/model · /api/samples\nPOST /api/score", "grey", fs=9.5)
box(ax, 0.24, 0.17, 0.52, 0.14, "CreditRiskEngine (loaded once at start-up)", "checks SHA-256 of bundle files + scikit-learn 1.9.0\n"
    "caches target-encoder lookups → ~0.7 s per request on Render", "model", fs=9.5)
arrow(ax, 0.48, 0.425, 0.52, 0.425)
arrow(ax, 0.64, 0.36, 0.55, 0.31)
arrow(ax, 0.835, 0.66, 0.6, 0.55)
ax.text(0.5, 0.06, "Free tier: the service sleeps after 15 idle minutes; the next visit wakes it in ~50 s.",
        ha="center", fontsize=9, color=MUTED)
fig.savefig(OUT / "deployment.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 4. ROC-AUC ladder
board = pd.read_csv(ROOT / "models" / "leaderboard.csv")
val = board[board["split"] == "validation"]
labels, a_vals, b_vals = [], [], []
for m in ["M0", "M1", "M2", "M3", "M4"]:
    labels.append(m)
    a_vals.append(float(val[(val["model"] == m) & (val["version"] == "A")]["roc_auc"].iloc[0]))
    b_vals.append(float(val[(val["model"] == m) & (val["version"] == "B")]["roc_auc"].iloc[0]))
extra = [("M3 B-mono", 0.7354), ("Final\n(no zip, OOF)", 0.7351), ("Final on\nTEST", 0.7465)]
fig, ax = plt.subplots(figsize=(8.3, 4.6))
import numpy as np  # noqa: E402
x = np.arange(len(labels))
ax.bar(x - 0.2, a_vals, 0.4, color="#c9d6ee", label="version A (with LendingClub grade / rate)")
ax.bar(x + 0.2, b_vals, 0.4, color="#3b6fd4", label="version B (deployable, no grade / rate)")
for i, (a, b) in enumerate(zip(a_vals, b_vals)):
    ax.text(i - 0.2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
    ax.text(i + 0.2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8, fontweight="bold")
x2 = np.arange(len(labels), len(labels) + len(extra))
colors = ["#e0a030", "#2e9b5b", "#c2412d"]
ax.bar(x2, [v for _, v in extra], 0.5, color=colors)
for i, (_, v) in zip(x2, extra):
    ax.text(i, v + 0.004, f"{v:.4f}" if v > 0.74 else f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(0.7148, color="#6b7280", ls="--", lw=1)
ax.text(-0.45, 0.705, "LendingClub interest rate on test: 0.715", fontsize=8, color=MUTED, ha="left", va="top")
ax.set_xticks(list(x) + list(x2), labels + [l for l, _ in extra], fontsize=9)
ax.set_ylim(0.45, 0.78)
ax.set_ylabel("ROC-AUC")
ax.set_title("ROC-AUC across the project (validation 2015 H1 unless marked TEST)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "auc_ladder.png", dpi=170)
plt.close(fig)

# ------------------------------------------------------------------ 5. SHAP waterfall for a real sample applicant
from src.engine import CreditRiskEngine  # noqa: E402
from src.modeling import reason_codes as rc  # noqa: E402
engine = CreditRiskEngine()
sample = json.loads((ROOT / "deploy" / "app" / "samples.json").read_text(encoding="utf-8"))[4]["application"]
app, missing, _ = engine._prepare(dict(sample), strict=False)
X = engine.transform([app])
contrib = engine.booster.predict(X, pred_contrib=True)
base = float(contrib[0, -1])
concepts = rc.concept_shap(contrib, engine.features).iloc[0].sort_values(key=abs, ascending=False)
top = concepts.head(8)
rest = concepts.iloc[8:].sum()
items = [(rc.describe(c, X.iloc[0]), v) for c, v in top.items()] + [(f"{len(concepts) - 8} other concepts", rest)]
fig, ax = plt.subplots(figsize=(8.3, 4.8))
cur = base
ys = list(range(len(items)))[::-1]
for y, (lab, v) in zip(ys, items):
    ax.barh(y, v, left=cur, color="#d4553b" if v > 0 else "#2e9b5b", height=0.6)
    ax.text(cur + v + (0.01 if v > 0 else -0.01), y, f"{v:+.3f}", va="center", ha="left" if v > 0 else "right", fontsize=8)
    cur += v
ax.set_yticks(ys, [l if len(l) < 60 else l[:57] + "…" for l, _ in items], fontsize=8.5)
ax.axvline(base, color=MUTED, ls=":", lw=1)
ax.axvline(cur, color=INK, ls="--", lw=1)
import math  # noqa: E402
ax.set_title(f"SHAP for one real applicant (small-business loan): start at the average log-odds {base:.2f} "
             f"(PD {1 / (1 + math.exp(-base)):.1%}),\nadd every push → {cur:.2f} (raw PD {1 / (1 + math.exp(-cur)):.1%}, "
             "before calibration)", loc="left", fontsize=9.5)
path = [base]
for _, v in items:
    path.append(path[-1] + v)
ax.set_xlim(min(path) - 0.12, max(path) + 0.12)
ax.set_xlabel("log-odds of default (red pushes risk up, green pushes it down)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "shap_waterfall.png", dpi=170)
plt.close(fig)
print("figures:", sorted(p.name for p in OUT.iterdir()))
