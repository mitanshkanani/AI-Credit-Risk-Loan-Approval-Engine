"""Assemble the Hugging Face Space folder (deploy/space/) with exactly the files the demo needs.

    python deploy/build_space.py                 # assemble + smoke-test in a fresh process
    python deploy/build_space.py --upload USER   # ...and upload to huggingface.co/spaces/USER/credit-risk-engine

The Space keeps the repo layout (src/, models/final/04_scoring_package/bundle/, deploy/app/), so the engine
finds its bundle exactly as it does locally. No data files are included.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "deploy" / "space"
FILES = [
    "src/__init__.py",
    "src/engine/__init__.py", "src/engine/scorer.py",
    "src/modeling/__init__.py", "src/modeling/decision.py", "src/modeling/reason_codes.py",
    "src/preprocessing_v2/__init__.py", "src/preprocessing_v2/config.py", "src/preprocessing_v2/features.py",
    "deploy/__init__.py", "deploy/app/__init__.py", "deploy/app/main.py", "deploy/app/samples.json",
    "deploy/app/model_card.json", "deploy/app/static/index.html",
    *[f"models/final/04_scoring_package/bundle/{f}" for f in
      ("preprocessor_v2.joblib", "final_model.txt", "calibrator.json", "decision_policy.json", "manifest.json")],
]
TOP = {"deploy/Dockerfile": "Dockerfile", "deploy/requirements.txt": "requirements.txt", "deploy/SPACE_README.md": "README.md"}

SMOKE = """
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from deploy.app.main import app
with TestClient(app) as c:
    assert c.get('/api/health').json()['status'] == 'ok'
    s = c.get('/api/samples').json()[0]
    r = c.post('/api/score', json=s['application']).json()
    assert r['decision'] == s['expected_decision'], r
    print('space smoke test OK:', r['decision'], r['probability_of_default'])
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", metavar="HF_USER", help="upload to huggingface.co/spaces/HF_USER/credit-risk-engine")
    args = parser.parse_args()

    if SPACE.exists():
        shutil.rmtree(SPACE)
    for rel in FILES:
        (SPACE / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, SPACE / rel)
    for src, dst in TOP.items():
        shutil.copy2(ROOT / src, SPACE / dst)
    size = sum(p.stat().st_size for p in SPACE.rglob("*") if p.is_file()) / 1e6
    print(f"assembled {SPACE} ({len(FILES) + len(TOP)} files, {size:.1f} MB)")

    # run the copied app from inside the Space folder, in a fresh process, so a missing file shows up here
    subprocess.run([sys.executable, "-c", SMOKE], cwd=SPACE, check=True)

    if args.upload:
        from huggingface_hub import HfApi
        api = HfApi()
        repo_id = f"{args.upload}/credit-risk-engine"
        api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(folder_path=str(SPACE), repo_id=repo_id, repo_type="space",
                          commit_message="Deploy credit-risk engine v1.0.0")
        print(f"uploaded: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
