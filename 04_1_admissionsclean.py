# 04_1_admissionsclean.py
# This file cleans and preprocesses the ADMISSIONS_sampled.parquet table.

import polars as pl
import os, pathlib
import datetime as _dt

PARQUET_DIR = r"H:\A\parquet files\sampled"
OUTPUT_DIR  = r"H:\A\parquet files\cleaned"

# helper functions for reporting
def _is_numeric(dtype_str: str) -> bool:
    return any(t in dtype_str for t in ["Int", "UInt", "Float"])

def _is_datetime(dtype_str: str) -> bool:
    return ("Datetime" in dtype_str) or ("Date" in dtype_str)

def _round2(x):
    if x is None:
        return None
    try:
        return round(float(x), 2)
    except Exception:
        return x  # non-numeric (e.g., datetime/string)

def _mode_of(df: pl.DataFrame, col: str):
    s = df.get_column(col)
    if s.len() == 0:
        return None
    # try the built-in mode first
    try:
        m = s.mode()
        if m.len() == 0:
            return None
        v = m[0]
    except Exception:
        # fallback via value_counts (drop nulls so mode isn't None unless it's all null)
        vc = s.drop_nulls().value_counts().sort("count", descending=True)
        if vc.height == 0:
            return None
        v = vc.row(0)[0]
    # pretty-print datetimes for CSV
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return v

def profile_dataframe(df: pl.DataFrame, table_name: str) -> pl.DataFrame:
    rows = df.height
    records = []
    for col, dtype in df.schema.items():
        dtype_str = str(dtype)
        nulls = df.select(pl.col(col).null_count().alias("n")).item()
        nunique = df.select(pl.col(col).n_unique().alias("n")).item()
        rec = {
            "column": col,
            "dtype": dtype_str,
            "rows": rows,
            "nulls": nulls,
            "n_unique": nunique,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "mode": _mode_of(df, col),
        }
        if _is_numeric(dtype_str):
            m = df.select([
                pl.col(col).mean().alias("mean"),
                pl.col(col).std().alias("std"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ]).row(0)
            rec["mean"], rec["std"], rec["min"], rec["max"] = map(_round2, m)
        elif _is_datetime(dtype_str):
            # format datetimes to readable strings
            mins = df.select(pl.col(col).min().alias("min")).row(0)[0]
            maxs = df.select(pl.col(col).max().alias("max")).row(0)[0]
            rec["min"] = None if mins is None else str(mins)
            rec["max"] = None if maxs is None else str(maxs)
        records.append(rec)
    return pl.DataFrame(records)

def compare_profiles(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    # prefix columns (no table names, no deltas)
    b = before.rename({k: f"{k}_before" for k in before.columns if k not in {"column"}})
    a = after.rename({k: f"{k}_after"  for k in after.columns  if k not in {"column"}})
    # outer join on column
    joined = b.join(a, on="column", how="outer")

    # reorder columns for readability
    cols = [
        "column",
        "dtype_before", "dtype_after",
        "rows_before", "rows_after",
        "nulls_before", "nulls_after",
        "n_unique_before", "n_unique_after",
        "mean_before", "mean_after",
        "std_before", "std_after",
        "min_before", "min_after",
        "max_before", "max_after",
        "mode_before", "mode_after",
    ]
    keep = [c for c in cols if c in joined.columns]
    return joined.select(keep)

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

admissions_before = pl.read_parquet(os.path.join(PARQUET_DIR, "ADMISSIONS_sampled.parquet"))
admissions = admissions_before.clone()  # work on a copy
print(f"Loaded ADMISSIONS: {admissions.shape[0]:,} rows, {admissions.shape[1]} columns")

# -- Cast IDs and force temporal columns to Datetime (works whether current type is date, datetime, or Utf8) --
time_cols = ["ADMITTIME","DISCHTIME","DEATHTIME","EDREGTIME","EDOUTTIME"]
admissions = admissions.with_columns(
    [
        pl.col("ROW_ID").cast(pl.Int64),
        pl.col("SUBJECT_ID").cast(pl.Int64),
        pl.col("HADM_ID").cast(pl.Int64),
        *[pl.col(c).cast(pl.Datetime).alias(c) for c in time_cols],
    ]
)

# -- Check for and drop duplicates which is unlikely --
admissions = admissions.unique(subset=["SUBJECT_ID", "HADM_ID"])
print(f"After dropping duplicates: {admissions.shape[0]:,}")

# -- Remove NEWBORNs and invalid times --
initial_count = admissions.shape[0]
admissions = admissions.filter(
    (pl.col("ADMISSION_TYPE") != "NEWBORN") &
    (pl.col("DISCHTIME") > pl.col("ADMITTIME"))
)
removed = initial_count - admissions.shape[0]
print(f"Removed {removed:,} rows (NEWBORNs / invalid timestamps)")

# -- Fill nulls (per-column) --
admissions = admissions.with_columns([
    pl.col("LANGUAGE").fill_null("UNKNOWN"),
    pl.col("RELIGION").fill_null("UNKNOWN"),
    pl.col("MARITAL_STATUS").fill_null("UNKNOWN"),
])

# -- LOS (days) --
admissions = admissions.with_columns(
    (pl.col("DISCHTIME") - pl.col("ADMITTIME")).dt.total_days().alias("LOS_DAYS")
)

# -- Summary --
summary = admissions.select([
    pl.len().alias("n_rows"),
    pl.col("LOS_DAYS").mean().round(2).alias("mean_LOS"),
    pl.col("ADMISSION_TYPE").n_unique().alias("n_types"),
    pl.col("INSURANCE").n_unique().alias("n_insurance"),
])
print("Summary stats:")
print(summary)

# ---- Before/After descriptive summary (rounded, with mode, no deltas) ----
prof_before = profile_dataframe(admissions_before, table_name="ADMISSIONS_sampled")
prof_after  = profile_dataframe(admissions,         table_name="ADMISSIONS_cleaned")
comparison  = compare_profiles(prof_before, prof_after)

summary_path = os.path.join(OUTPUT_DIR, "ADMISSIONS_cleaning_summary.csv")
comparison.write_csv(summary_path)
print(f"[REPORT] Wrote before/after summary → {summary_path}")

print(
    comparison.head(12)
    .to_pandas()  # nicer console table rendering
)

# -- Save --
out_path = os.path.join(OUTPUT_DIR, "ADMISSIONS_cleaned.parquet")
admissions.write_parquet(out_path, compression="zstd")
print(f"Saved cleaned ADMISSIONS → {out_path}")
print("Done.")