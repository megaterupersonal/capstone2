# 05_07a_audit.py
# Purpose: Frequency of each ITEMID in LABEVENTS_cleaned.parquet
# Detect which ITEMIDs contain non-numeric VALUEs
# Report how many such rows exist per ITEMID (and % of that item’s rows)
# Show the mode text (most frequent string value)
# Output a single CSV for review: LABEVENTS_itemid_text_audit.csv
# Keep your existing frequency report untouched
# Output:  features/LABEVENTS_itemid_frequency.csv

import polars as pl
import os, pathlib
import re

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

LABEVENTS_PATH = os.path.join(CLEAN_DIR, "LABEVENTS_cleaned.parquet")
OUT_FREQ = os.path.join(FEAT_DIR, "LABEVENTS_itemid_frequency.csv")
OUT_TEXT = os.path.join(FEAT_DIR, "LABEVENTS_itemid_text_audit.csv")

# ---------- Load ----------
le = pl.read_parquet(LABEVENTS_PATH)

# ---------- Sanity ----------
for c in ["ITEMID", "HADM_ID", "VALUE"]:
    if c not in le.columns:
        raise ValueError(f"LABEVENTS_cleaned.parquet missing '{c}'")

n_total = le.height
n_total_hadm = le.select(pl.n_unique("HADM_ID")).item()

# ---------- 1) Frequency by admissions (unchanged) ----------
freq = (
    le.group_by("ITEMID")
      .agg(pl.n_unique("HADM_ID").alias("n_hadm"))
      .with_columns(
          (pl.col("n_hadm") / n_total_hadm * 100).round(4).alias("percent_of_admissions")
      )
      .sort("n_hadm", descending=True)
)
freq.write_csv(OUT_FREQ)
print(f"[done] Frequency table saved to {OUT_FREQ}")

# ---------- 2) Text-value audit with admissions coverage ----------
# Helper: detect non-numeric VALUEs (treat '>5', '<3', '5.6' as numeric)
def is_non_numeric(s: str) -> bool:
    if s is None:
        return False
    s = s.strip()
    if s == "":
        return False
    # numeric patterns like "5", "5.6", ">5", "< 3"
    if re.match(r"^[<>]?\s*\d+(\.\d+)?$", s):
        return False
    return True

le = le.with_columns(
    pl.col("VALUE").map_elements(is_non_numeric, return_dtype=pl.Boolean).alias("IS_TEXT_VALUE")
)

# counts per ITEMID
total_rows_by_item = le.group_by("ITEMID").agg(pl.len().alias("n_total_rows"))
text_rows = le.filter(pl.col("IS_TEXT_VALUE") == True)

# text-row stats
text_row_stats = (
    text_rows.group_by("ITEMID")
      .agg([
          pl.len().alias("n_text_rows"),
          pl.col("VALUE").drop_nulls().mode().first().alias("mode_text"),
          pl.n_unique("HADM_ID").alias("n_hadm_with_text")
      ])
)

# overall admissions coverage per ITEMID (for join)
adm_coverage = le.group_by("ITEMID").agg(pl.n_unique("HADM_ID").alias("n_hadm_any"))

# assemble audit
text_audit = (
    total_rows_by_item
    .join(text_row_stats, on="ITEMID", how="left")
    .join(adm_coverage, on="ITEMID", how="left")
    .with_columns([
        pl.col("n_text_rows").fill_null(0),
        pl.col("mode_text").fill_null(""),
        pl.col("n_hadm_with_text").fill_null(0),
        (pl.col("n_text_rows") / pl.col("n_total_rows") * 100).round(2).alias("pct_text_rows"),
        (pl.col("n_hadm_any") / n_total_hadm * 100).round(4).alias("percent_of_admissions"),
        (pl.when(pl.col("n_hadm_any") > 0)
   .then(pl.col("n_hadm_with_text") / pl.col("n_hadm_any") * 100)
   .otherwise(None)
   .round(2)
   .alias("pct_admissions_with_text"))
    ])
)


for c in ["LABEL", "FLUID"]:
    if c in le.columns:
        meta = le.select(["ITEMID", c]).unique()
        text_audit = text_audit.join(meta, on="ITEMID", how="left")

text_audit = (
    text_audit
    .select([
        "ITEMID", "LABEL", "FLUID",
        "n_total_rows", "n_text_rows", "pct_text_rows",
        "n_hadm_any", "percent_of_admissions",
        "n_hadm_with_text", "pct_admissions_with_text",  # <- fixed name here
        "mode_text",
    ])
    .sort(["n_text_rows", "n_total_rows"], descending=[True, True])
)

text_audit.write_csv(OUT_TEXT)
print(f"[done] Text-value audit saved to {OUT_TEXT}")
print(text_audit.head(20))
print(f"\nTotal rows in LABEVENTS: {n_total:,} | Unique admissions: {n_total_hadm:,}")