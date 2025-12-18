# 04_6_microbiologyeventsclean.py
# This file cleans and preprocesses the MICROBIOLOGYEVENTS_sampled.parquet table.

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

# ---------- Load ----------
micro_before = pl.read_parquet(os.path.join(PARQUET_DIR, "MICROBIOLOGYEVENTS_sampled.parquet"))
micro = micro_before.clone()
print(f"Loaded MICROBIOLOGYEVENTS: {micro.shape[0]:,} rows, {micro.shape[1]} columns")

# ---------- Types ----------
micro = micro.with_columns([
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("CHARTDATE").cast(pl.Datetime),
    pl.col("CHARTTIME").cast(pl.Datetime),
    pl.col("ORG_NAME").cast(pl.Utf8),
    pl.col("AB_NAME").cast(pl.Utf8),
    pl.col("INTERPRETATION").cast(pl.Utf8),
])

# ---------- Drop rows with no biological signal ----------
# Keep a row if ANY of these carry information (we won’t standardize content here)
before = micro.height
micro = micro.filter(
    (pl.col("ORG_NAME").is_not_null()) |
    (pl.col("AB_NAME").is_not_null()) |
    (pl.col("INTERPRETATION").is_not_null())
)
print(f"Removed {before - micro.height:,} rows with no organism/antibiotic/interpretation.")

# ---------- Exact-duplicate removal only (no key-based collapsing) ----------
before = micro.height
micro = micro.unique(maintain_order=True)
print(f"Removed {before - micro.height:,} exact duplicate rows.")

# ---------- Admission-level infection flag ----------
# We only use ORG_NAME to detect a positive organism.
# Treat obvious negatives as non-infection; anything else non-null counts as infection.
NEG_ORG_PATTERNS = {
    "NO GROWTH",
    "NORMAL FLORA",
    "NORMAL SKIN FLORA",
    "NEGATIVE",
    "NONE SEEN",
    "NONE ISOLATED",
}

# ---------- Improved negative/neutral detection ----------
# Normalize once
def _norm(s: pl.Expr) -> pl.Expr:
    return (
        s.cast(pl.Utf8)
         .str.to_uppercase()
         .str.strip_chars()
         .str.replace_all(r"\s+", " ")           # collapse internal whitespace
    )

org_txt = _norm(pl.col("ORG_NAME"))
interp_txt = _norm(pl.col("INTERPRETATION"))
ab_txt = _norm(pl.col("AB_NAME"))

# Scan ORG_NAME first; if empty, consider INTERPRETATION for negative/neutral cues.
# (This keeps "positive-by-ORG" behaviour, but avoids false positives on null ORG_NAME.)

NEG_RE = (
    r"(?x)"  # verbose mode
    r"(?:\bNO\s+(?:GROWTH(?:\s+TO\s+DATE)?(?:\s+AT\s*\d+\s*H)?|"
    r"ORGANISMS?(?:\s+SEEN|\s+ISOLATED)?|"
    r"BACTERIA(?:\s+SEEN)?|YEAST(?:\s+SEEN)?|PATHOGENS?)\b"
    r"|^NGTD$|^NG$|^NO\s*G$|"
    r"\bNEGATIVE\b|"
    r"\bNOT\s+DETECTED\b|"
    r"\bNONE\s+(?:SEEN|ISOLATED)\b|"
    r"\bNO\s+SIGNIFICANT\s+GROWTH\b|"
    r"\bSTERILE\b|"
    r"\bNORMAL(?:\s+SKIN)?\s+FLORA\b|"
    r"\bMIXED(?:\s+SKIN)?\s+FLORA\b|"
    r"(?:\bPROBABLE\b\s+)?\bCONTAMINANT\b|"
    r"\bCOMMENSAL\b|"
    r"\bCOLONI[ZS]ER\b)"
)

NEUTRAL_RE = (
    r"(?x)"
    r"\bPENDING\b|"
    r"\bTO\s+FOLLOW\b|"
    r"\bREINCUBAT(?:E|ED|ION)\b|"
    r"\bREPEAT\s+CULTURE\b|"
    r"\bSEE\s+(?:NOTE|COMMENT|REPORT)\b|"
    r"\bINSUFFICIENT\s+SAMPLE\b|"
    r"\bSPECIMEN\s+REJECTED\b|"
    r"\bHELD\s+FOR\s+ADDITIONAL\s+TESTS?\b|"
    r"\bCORRECTED\s+REPORT\b"
)

# Where to look for negative/neutral language:
neg_hit = (
    org_txt.fill_null("")
           .str.contains(NEG_RE)
    |
    (
        (org_txt.is_null() | (org_txt == "")) &
        interp_txt.fill_null("").str.contains(NEG_RE)
    )
)

neutral_hit = (
    org_txt.fill_null("").str.contains(NEUTRAL_RE)
    |
    (
        (org_txt.is_null() | (org_txt == "")) &
        interp_txt.fill_null("").str.contains(NEUTRAL_RE)
    )
)

# capture a short reason label for auditing
neg_reason = (
    pl.when(neg_hit).then(pl.lit("NEG_PHRASE"))
     .when(neutral_hit).then(pl.lit("NEUTRAL_PHRASE"))
     .otherwise(pl.lit(None))
     .alias("NEG_REASON")
)

# ---------- Admission-level infection summary ----------
micro_flagged = micro.with_columns([
    neg_reason,
    pl.when(neg_hit | neutral_hit | org_txt.is_null() | (org_txt == ""))
      .then(0)
      .otherwise(1)
      .alias("IS_POS_ORG")
])

infection_by_hadm = (
    micro_flagged
      .group_by("HADM_ID")
      .agg([
          pl.len().alias("MICRO_SRC_N_ROWS"),                  # rows that survived your cleaning
          pl.col("IS_POS_ORG").sum().alias("MICRO_SRC_N_POS"), # number of positive rows
          pl.col("IS_POS_ORG").max().alias("HAS_INFECTION"),   # 0/1
      ])
      .with_columns([
          pl.col("HAS_INFECTION").cast(pl.Int8),
          pl.when(pl.col("MICRO_SRC_N_ROWS") > 0)
            .then((pl.col("MICRO_SRC_N_POS") / pl.col("MICRO_SRC_N_ROWS")).round(3))
            .otherwise(None)
            .alias("MICRO_SRC_POS_RATE"),
      ])
      .sort("HADM_ID")
)

out_inf = os.path.join(OUTPUT_DIR, "MICROBIOLOGY_infection_by_hadm.parquet")
infection_by_hadm.write_parquet(out_inf, compression="zstd")


# Before/after profile on the *raw* micro table ----------
prof_before = profile_dataframe(micro_before, table_name="MICROBIOLOGYEVENTS_sampled")
prof_after  = profile_dataframe(micro,        table_name="MICROBIOLOGYEVENTS_minimal")
comparison  = compare_profiles(prof_before, prof_after)
comparison.write_csv(os.path.join(OUTPUT_DIR, "MICROBIOLOGYEVENTS_cleaning_summary.csv"))
print("[REPORT] Wrote before/after summary → MICROBIOLOGYEVENTS_cleaning_summary.csv")

# ---------- Save only the compact, join-ready file ----------
out_inf = os.path.join(OUTPUT_DIR, "MICROBIOLOGY_infection_by_hadm.parquet")
infection_by_hadm.write_parquet(out_inf, compression="zstd")
print(f"[OK] Wrote infection flag → {out_inf}")

# Quick sanity print
n_hadm_total = micro.select(pl.col("HADM_ID").n_unique()).item()
n_hadm_flag  = infection_by_hadm.select(pl.col("HAS_INFECTION").sum()).item()
print(f"[INFO] Admissions with any positive organism: {n_hadm_flag:,} / {n_hadm_total:,} "
      f"({(n_hadm_flag/n_hadm_total*100):.2f}%)")
# n_hadm_total should measure how many unique HADM_IDs were in the original MICROBIOLOGYEVENTS_sampled.parquet
# n_hadm_flag should measure how many of those had HAS_INFECTION = 1
# The ratio gives the % of admissions with detected infections

print("Done.")
