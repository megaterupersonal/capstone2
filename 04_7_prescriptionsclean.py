# 04_7_prescriptionsclean.py
# This file cleans and preprocesses the PRESCRIPTIONS_sampled.parquet table.

import polars as pl
import os, pathlib, re
import datetime as _dt

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

PARQUET_DIR = r"H:\A\parquet files\sampled"
OUTPUT_DIR  = r"H:\A\parquet files\cleaned"
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ---- Load ----
pres_before = pl.read_parquet(os.path.join(PARQUET_DIR, "PRESCRIPTIONS_sampled.parquet"))
pres = pres_before.clone()
print(f"Loaded PRESCRIPTIONS: {pres.shape[0]:,} rows, {pres.shape[1]} columns")

# ---- Type casting ----
pres = pres.with_columns([
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("DRUG").cast(pl.Utf8),
    pl.col("STARTDATE").cast(pl.Datetime),
    pl.col("ENDDATE").cast(pl.Datetime),
    pl.col("DOSE_VAL_RX").cast(pl.Utf8),
    pl.col("DOSE_UNIT_RX").cast(pl.Utf8),
])

# ---- Drop missing drug/dose info ----
before = pres.height
pres = pres.filter(
    (pl.col("DRUG").is_not_null()) &
    (pl.col("DRUG").str.strip_chars() != "")
)
print(f"Removed {before - pres.height:,} rows with missing DRUG info.")

# ---- Fix nonsensical dates ----
before = pres.height
pres = pres.filter(
    (pl.col("ENDDATE").is_null()) | (pl.col("ENDDATE") >= pl.col("STARTDATE"))
)
print(f"Removed {before - pres.height:,} rows with invalid dates (END < START).")

# ---- Clean drug names ----
pres = pres.with_columns([
    pl.col("DRUG").str.strip_chars().str.to_titlecase().alias("DRUG"),
    pl.col("DOSE_UNIT_RX").str.strip_chars().str.to_uppercase().alias("DOSE_UNIT_RX")
])

# ---- Extract numeric dose (lower bound) ----
# Convert "325-650" → 325.0, keep as float
def _extract_lower(val: str):
    if val is None:
        return None
    m = re.match(r"(\d+(\.\d+)?)(-|/)?", val)
    return float(m.group(1)) if m else None

pres = pres.with_columns([
    pl.col("DOSE_VAL_RX").map_elements(_extract_lower, return_dtype=pl.Float64).alias("DOSE_VAL_NUM")
])

# ---- Drop exact duplicates ----
before = pres.height
pres = pres.unique(maintain_order=True)
print(f"Removed {before - pres.height:,} exact duplicate rows.")

# ---- Calculate duration in days (optional) ----
pres = pres.with_columns([
    ((pl.col("ENDDATE") - pl.col("STARTDATE")).dt.total_days())
      .alias("DURATION_DAYS")
])

# ---- Flag chronic (>=5 days) vs acute ----
pres = pres.with_columns([
    pl.when(pl.col("DURATION_DAYS") >= 5).then(1).otherwise(0).alias("IS_CHRONIC")
])

# ---- Profiling summary ----
prof_before = profile_dataframe(pres_before, table_name="PRESCRIPTIONS_sampled")
prof_after  = profile_dataframe(pres,        table_name="PRESCRIPTIONS_cleaned")
comparison  = compare_profiles(prof_before, prof_after)

summary_path = os.path.join(OUTPUT_DIR, "PRESCRIPTIONS_cleaning_summary.csv")
comparison.write_csv(summary_path)
print(f"[REPORT] Wrote before/after summary → {summary_path}")

# ---- Save cleaned file ----
out_path = os.path.join(OUTPUT_DIR, "PRESCRIPTIONS_cleaned.parquet")
pres.write_parquet(out_path, compression="zstd")
print(f"Saved cleaned PRESCRIPTIONS → {out_path}")
print("Done.")
