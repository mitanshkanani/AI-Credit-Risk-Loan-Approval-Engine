"""Assemble the scoring bundle: every file the engine needs, in one folder, with a manifest.

    python models/final/04_scoring_package/build_bundle.py

bundle/
  preprocessor_v2.joblib   fitted preprocessing pipeline (Preprocessing V2, TRAIN only)
  final_model.txt          LightGBM M3-B-mono-nozip (Stage 2)
  calibrator.json          Platt calibration (Stage 2)
  decision_policy.json     risk bands, thresholds, loss model (Stage 3)
  manifest.json            checksums, library versions, features, input fields

The bundle is the single source for the scoring package (Stage 4), the final test (Stage 5)
and the live demo (Stage 6).
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import lightgbm as lgb  # noqa: E402
import sklearn  # noqa: E402

from src.engine.scorer import raw_field_of  # noqa: E402

BUNDLE = Path(__file__).resolve().parent / "bundle"
STAGE2 = ROOT / "models" / "final" / "02_reason_codes_fairness" / "results"
SOURCES = {
    "preprocessor": ROOT / "artifacts_v2" / "preprocessor_v2.joblib",
    "model": STAGE2 / "final_model.txt",
    "calibrator": STAGE2 / "calibrator.json",
    "policy": ROOT / "models" / "final" / "03_decision_layer" / "results" / "decision_policy.json",
}

BUNDLE.mkdir(exist_ok=True)
for source in SOURCES.values():
    shutil.copy2(source, BUNDLE / source.name)

choice = json.loads((STAGE2 / "final_model_choice.json").read_text(encoding="utf-8"))
metadata = json.loads((ROOT / "artifacts_v2" / "metadata_v2.json").read_text(encoding="utf-8"))
features = choice["features"]
preprocessor = joblib.load(SOURCES["preprocessor"])
raw_fields = list(preprocessor.named_steps["engineer"].input_columns_)
model_fields = list(dict.fromkeys(raw_field_of(f, raw_fields) for f in features))
booster = lgb.Booster(model_file=str(SOURCES["model"]))
assert booster.num_feature() == len(features) == choice["n_features"]
assert [c.replace(" ", "_") for c in features] == booster.feature_name(), "feature order differs from the model"

manifest = {
    "model_name": choice["chosen_model"],
    "version": "1.0.0",
    "description": "LendingClub accepted-loan credit-risk engine: PD, risk band, decision, expected loss, reasons",
    "files": {k: v.name for k, v in SOURCES.items()},
    "sha256": {v.name: hashlib.sha256((BUNDLE / v.name).read_bytes()).hexdigest() for v in SOURCES.values()},
    "environment": {
        "scikit_learn": metadata["environment"]["scikit_learn"],
        "python_fitted": metadata["environment"]["python"],
        "lightgbm_trained": choice["environment"]["lightgbm"],
        "lightgbm_verified": lgb.__version__,
    },
    "features": features,
    "model_input_fields": model_fields,
    "ignored_raw_fields": [f for f in raw_fields if f not in model_fields and f != "issue_d"],
    "application_month_field": "issue_d",
    "training_window": "2012-08..2014-12 (train), 2015 H1 (validation: early stopping, calibration, thresholds)",
}
assert sklearn.__version__ == manifest["environment"]["scikit_learn"], "build the bundle with the scikit-learn that fitted the preprocessor"
(BUNDLE / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"bundle: {BUNDLE}")
print(f"  {len(features)} features from {len(model_fields)} raw input fields; {len(manifest['ignored_raw_fields'])} raw fields ignored")
for name in manifest["sha256"]:
    print(f"  {name:28s} {(BUNDLE / name).stat().st_size / 1e6:6.2f} MB")
