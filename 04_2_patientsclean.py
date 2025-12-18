# 04_2_patientsclean.py
# This file cleans and preprocesses the PATIENTS_sampled.parquet table.
#Summary helpers always come first and adapted from previous files

import polars as pl
import os, pathlib
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
patients_before = pl.read_parquet(os.path.join(PARQUET_DIR, "PATIENTS_sampled.parquet"))
patients = patients_before.clone()
print(f"Loaded PATIENTS: {patients.shape[0]:,} rows, {patients.shape[1]} columns")

# ---- Cast dtypes ----
patients = patients.with_columns([
    pl.col("ROW_ID").cast(pl.Int64),
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("EXPIRE_FLAG").cast(pl.Int64),
    pl.col("GENDER").cast(pl.Categorical),
    pl.col("DOB").cast(pl.Datetime),
    pl.col("DOD").cast(pl.Datetime),
    pl.col("DOD_HOSP").cast(pl.Datetime),
    pl.col("DOD_SSN").cast(pl.Datetime),
])

# ---- Drop duplicates ----
patients = patients.unique(subset=["SUBJECT_ID"])
print(f"After dropping duplicates: {patients.shape[0]:,}")

# ---- Check invalid date order (DOB > any DOD) ----
invalid_dates = patients.filter(
    (pl.col("DOD") < pl.col("DOB")) |
    (pl.col("DOD_HOSP") < pl.col("DOB")) |
    (pl.col("DOD_SSN") < pl.col("DOB"))
)
if invalid_dates.height > 0:
    print(f"[WARN] Found {invalid_dates.height} records with invalid date order; removing them.")
    patients = patients.join(invalid_dates, on="SUBJECT_ID", how="anti")

# ---- Combine DOD fields into DOD_FINAL ----
patients = patients.with_columns([
    pl.coalesce(["DOD", "DOD_HOSP", "DOD_SSN"]).alias("DOD_FINAL")
])

# ---- Drop redundant DOD columns ----
patients = patients.drop(["DOD", "DOD_HOSP", "DOD_SSN"])
print("Dropped redundant DOD columns (DOD, DOD_HOSP, DOD_SSN).")

# ---- Fill missing EXPIRE_FLAG ----
patients = patients.with_columns([
    pl.col("EXPIRE_FLAG").fill_null(0)
])

# ---- Cleaning summary ----
prof_before = profile_dataframe(patients_before, table_name="PATIENTS_sampled")
prof_after  = profile_dataframe(patients,         table_name="PATIENTS_cleaned")
comparison  = compare_profiles(prof_before, prof_after)

summary_path = os.path.join(OUTPUT_DIR, "PATIENTS_cleaning_summary.csv")
comparison.write_csv(summary_path)
print(f"[REPORT] Wrote before/after summary → {summary_path}")

# ---- Save cleaned ----
out_path = os.path.join(OUTPUT_DIR, "PATIENTS_cleaned.parquet")
patients.write_parquet(out_path, compression="zstd")
print(f"Saved cleaned PATIENTS → {out_path}")
print("Done.")
