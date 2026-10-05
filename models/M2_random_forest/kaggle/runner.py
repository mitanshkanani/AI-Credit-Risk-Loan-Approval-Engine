"""Kaggle runner for M2: rebuild the repo layout and execute the M2 notebook.

Runs as a private Kaggle script kernel (see kernel-metadata.json). The input dataset
`mitanshkanani/credit-risk-m2-inputs` contains train/validation data (no test), the
shared evaluation module, the leaderboard, M1-B validation predictions and the notebook.
The executed notebook and results/ end up under /kaggle/working/repo/models/M2_random_forest/.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT = next(Path("/kaggle/input").rglob("M2_random_forest.ipynb")).parent
REPO = Path("/kaggle/working/repo")
print("Input dataset folder:", INPUT)
print("Files:", sorted(p.name for p in INPUT.iterdir()))

modeling = REPO / "src" / "modeling"
modeling.mkdir(parents=True, exist_ok=True)
(REPO / "src" / "__init__.py").touch()
(modeling / "__init__.py").touch()
shutil.copy(INPUT / "evaluation.py", modeling / "evaluation.py")

m1_results = REPO / "models" / "M1_logistic_regression" / "results"
m1_results.mkdir(parents=True, exist_ok=True)
shutil.copy(INPUT / "leaderboard.csv", REPO / "models" / "leaderboard.csv")
shutil.copy(INPUT / "m1_B_validation_pd.parquet", m1_results / "m1_B_validation_pd.parquet")

notebook_dir = REPO / "models" / "M2_random_forest"
notebook_dir.mkdir(parents=True, exist_ok=True)
notebook = notebook_dir / "M2_random_forest.ipynb"
shutil.copy(INPUT / "M2_random_forest.ipynb", notebook)

print(f"CPU cores: {os.cpu_count()} | executing {notebook}", flush=True)
start = time.time()
subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
     "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3", str(notebook)],
    check=True,
)
print(f"Notebook executed in {(time.time() - start) / 60:.1f} minutes")
print("Outputs:", sorted(str(p.relative_to(REPO)) for p in notebook_dir.rglob("*") if p.is_file()))
