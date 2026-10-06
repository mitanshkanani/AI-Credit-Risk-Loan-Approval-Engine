"""Extract the repayment / recovery columns needed for LGD from the raw LendingClub CSV.

    python models/final/03_decision_layer/extract_loss_table.py

These columns are post-origination outcomes, so they were (correctly) removed from the model
features. Stage 3 needs them only to measure loss given default (LGD) and exposure at default
(EAD). Only TRAIN and VALIDATION loans are kept; the sealed test split is not touched.
The raw CSV is read in chunks so the laptop's free memory is never an issue.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "accepted_2007_to_2018Q4.csv"
OUT = Path(__file__).resolve().parent / "results" / "loss_table.parquet"
COLUMNS = ["id", "loan_amnt", "funded_amnt", "term", "int_rate", "loan_status", "out_prncp",
           "total_pymnt", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
           "recoveries", "collection_recovery_fee"]

split_of = {}
for split in ("train", "validation"):
    ids = pd.read_parquet(ROOT / "final_preprocessed_data_v2" / f"y_{split}.parquet", columns=["id"])["id"]
    split_of.update(dict.fromkeys(ids.astype(str), split))

parts = []
for chunk in pd.read_csv(RAW, usecols=COLUMNS, dtype={"id": str}, chunksize=200_000, low_memory=False):
    chunk = chunk[chunk["id"].isin(split_of.keys())]
    parts.append(chunk)
table = pd.concat(parts, ignore_index=True)
table.insert(1, "split", table["id"].map(split_of))
table["term_months"] = table.pop("term").str.extract(r"(\d+)")[0].astype(int)
table.to_parquet(OUT, index=False)

expected = len(split_of)
print(f"kept {len(table):,} of {expected:,} train+validation loans -> {OUT}")
print(table.groupby("split").size())
assert len(table) == expected and table["id"].is_unique, "loan ids missing or duplicated"
