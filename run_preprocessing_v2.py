"""Run accepted-loan preprocessing V2 end to end.

    python run_preprocessing_v2.py

Writes only to final_preprocessed_data_v2/ and artifacts_v2/.
"""

import sys

from src.preprocessing_v2.run import run_all

if __name__ == "__main__":
    ctx = run_all()
    print("\nPREPROCESSING V2:", "COMPLETE" if ctx["complete"] else "NOT COMPLETE")
    sys.exit(0 if ctx["complete"] else 1)
