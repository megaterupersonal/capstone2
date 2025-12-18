# 04_5_labeventsclean.py
# This file cleans and preprocesses the LABEVENTS_sampled.parquet table.

import polars as pl
import os, pathlib
import datetime as _dt
import re  

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
labs_before = pl.read_parquet(os.path.join(PARQUET_DIR, "LABEVENTS_sampled.parquet"))
labs = labs_before.clone()
print(f"Loaded LABEVENTS: {labs.shape[0]:,} rows, {labs.shape[1]} columns")

# ---- Type casting ----
labs = labs.with_columns([
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("ITEMID").cast(pl.Int64),
    pl.col("CHARTTIME").cast(pl.Datetime),
    pl.col("VALUENUM").cast(pl.Float64)
])

# ---- Drop null key fields ----
before = labs.shape[0]
labs = labs.filter(
    (pl.col("SUBJECT_ID").is_not_null()) &
    (pl.col("HADM_ID").is_not_null()) &
    (pl.col("ITEMID").is_not_null()) &
    (pl.col("CHARTTIME").is_not_null())
)
print(f"Removed {before - labs.shape[0]:,} rows with null key fields.")

# ---- Drop rows with both VALUE and VALUENUM null ----
before = labs.shape[0]
labs = labs.filter(
    (pl.col("VALUE").is_not_null()) | (pl.col("VALUENUM").is_not_null())
)
print(f"Removed {before - labs.shape[0]:,} rows with missing VALUE and VALUENUM.")

# ---- Round VALUENUM to 2 dp ----
labs = labs.with_columns([
    pl.when(pl.col("VALUENUM").is_not_null())
      .then(pl.col("VALUENUM").round(2))
      .otherwise(None)
      .alias("VALUENUM")
])

# ---- Standardize FLAG ----
flag_str = pl.col("FLAG").cast(pl.Utf8).str.to_lowercase()

labs = labs.with_columns([
    pl.when(flag_str == pl.lit("abnormal"))
      .then(pl.lit("abnormal"))
      .when(flag_str.is_null() | (flag_str == pl.lit("")))
      .then(pl.lit("unknown"))
      .otherwise(pl.lit("unknown"))  # e.g., "delta"
      .alias("FLAG_CLEAN"),
    pl.when(flag_str == pl.lit("abnormal"))
      .then(1)
      .otherwise(0)
      .alias("IS_ABNORMAL")
])

# ---- Clean text columns (VALUEUOM / LABEL / CATEGORY / FLUID) ----
text_cols = ["VALUEUOM", "LABEL", "CATEGORY", "FLUID"]
labs = labs.with_columns([
    pl.col(c).cast(pl.Utf8).str.strip_chars().str.to_titlecase().alias(c)
    for c in text_cols
])

# ---- Keep ITEMIDs with ≥10% admissions coverage ----
COVERAGE_CUTOFF = 10.0  # percent
n_adm = labs.select(pl.n_unique("HADM_ID")).item()
keep_itemids = (
    labs.group_by("ITEMID")
        .agg(pl.n_unique("HADM_ID").alias("n_hadm"))
        .with_columns((pl.col("n_hadm") / n_adm * 100).alias("pct_adm"))
        .filter(pl.col("pct_adm") >= COVERAGE_CUTOFF)
        .select("ITEMID").to_series().to_list()
)
before = labs.height
labs = labs.filter(pl.col("ITEMID").is_in(keep_itemids))
print(f"Applied 10% admissions filter: kept {len(keep_itemids)} ITEMIDs; "
      f"removed {before - labs.height:,} rows.")

# ---- Handle text VALUEs: censored values + admin errors ----
NUMERIC_TOKEN = r"^[<>]?\s*\d+(\.\d+)?$"
LT_RX = r"(?:^|[\s(])(?:(?:<=)|<|≤|less\s+than)\s*(\d+(?:\.\d+)?)"
GT_RX = r"(?:^|[\s(])(?:(?:>=)|>|≥|greater\s+than)\s*(\d+(?:\.\d+)?)"
ERROR_PATTERNS = [
    r"\berror\b",
    r"\bunable to report\b",
    r"\btest not resulted\b",
    r"\bdisregard previous\b",
    r"\bnetwork failure\b",
    r"\bcancel+ed\b",
    r"\binvalid\b",
]

def _is_error(v: str) -> bool:
    if v is None:
        return False
    t = v.strip().lower()
    for rx in ERROR_PATTERNS:
        if re.search(rx, t):
            return True
    return False

# STEP A: normalized helper
labs = labs.with_columns(
    pl.col("VALUE").cast(pl.Utf8).str.strip_chars().str.to_lowercase().alias("_vlow")
)

# STEP B: basic flags & thresholds (no cross-dependencies yet)
labs = labs.with_columns([
    pl.col("VALUE").map_elements(_is_error, return_dtype=pl.Boolean).alias("IS_ERROR_TEXT"),
    pl.col("_vlow").str.contains(LT_RX).fill_null(False).alias("_has_lt"),
    pl.col("_vlow").str.contains(GT_RX).fill_null(False).alias("_has_gt"),
    pl.col("_vlow").str.extract(LT_RX, group_index=1).cast(pl.Float64).alias("_lt_thr"),
    pl.col("_vlow").str.extract(GT_RX, group_index=1).cast(pl.Float64).alias("_gt_thr"),
])


# STEP C: generic text flag (can now reference _has_lt/_has_gt safely)
labs = labs.with_columns(
    pl.when(pl.col("VALUE").is_not_null())
      .then(
          (~pl.col("_vlow").str.contains(NUMERIC_TOKEN).fill_null(False))
          & ~pl.col("_has_lt") & ~pl.col("_has_gt")
      )
      .otherwise(False)
      .alias("IS_TEXT_VALUE")
)

# ----- STEP D: final numeric & censor flags (keep as you already have above) -----
# labs = labs.with_columns([... VAL_FINAL, LT_FLAG, GT_FLAG ...])

# ----- STEP D.1: strengthened inequality extraction (allow sign & commas) -----
# Example matches: "< 1,234.5", "greater than -0.02", "≤ 3", ">= -1.5"
NUM_WITH_COMMAS = r"(-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)"

LT_RX2 = rf"(?:^|[\s(])(?:(?:<=)|<|≤|less\s+than)\s*{NUM_WITH_COMMAS}"
GT_RX2 = rf"(?:^|[\s(])(?:(?:>=)|>|≥|greater\s+than)\s*{NUM_WITH_COMMAS}"

labs = labs.with_columns([
    pl.col("_vlow").str.extract(LT_RX2, group_index=1)
        .str.replace_all(",", "")
        .cast(pl.Float64)
        .alias("_lt_thr"),
    pl.col("_vlow").str.extract(GT_RX2, group_index=1)
        .str.replace_all(",", "")
        .cast(pl.Float64)
        .alias("_gt_thr"),
])

# ----- STEP D.2: apply epsilon so the numeric reflects the strict inequality -----
# round to 2 dp later
DECIMALS = 2
EPS = 10 ** (-DECIMALS)    # 0.01
labs = labs.with_columns([
    (pl.col("_gt_thr") + EPS).alias("_gt_adj"),
    (pl.col("_lt_thr") - EPS).alias("_lt_adj"),
])

# ----- STEP D.3: compose VAL_FINAL with precedence -----
labs = labs.with_columns([
    pl.when(pl.col("VALUENUM").is_not_null())          # numeric value wins
      .then(pl.col("VALUENUM"))
      .when(pl.col("_has_gt") & pl.col("_gt_adj").is_not_null())
      .then(pl.col("_gt_adj"))                         # > t  → t + 0.01
      .when(pl.col("_has_lt") & pl.col("_lt_adj").is_not_null())
      .then(pl.col("_lt_adj"))                         # < t  → t - 0.01
      .otherwise(None)
      .alias("VAL_FINAL")
])


# ----- STEP E: report inequality conversions -----
# Count candidates (non-error rows with detected inequality) and those actually converted (threshold present)
ineq_stats = labs.select([
    ((pl.col("_has_lt")) & (~pl.col("IS_ERROR_TEXT"))).sum().alias("n_lt_candidates"),
    ((pl.col("_has_gt")) & (~pl.col("IS_ERROR_TEXT"))).sum().alias("n_gt_candidates"),
    ((pl.col("_has_lt")) & (~pl.col("IS_ERROR_TEXT")) & pl.col("_lt_thr").is_not_null()).sum().alias("n_lt_converted"),
    ((pl.col("_has_gt")) & (~pl.col("IS_ERROR_TEXT")) & pl.col("_gt_thr").is_not_null()).sum().alias("n_gt_converted"),
]).row(0)
n_lt_cand, n_gt_cand, n_lt_conv, n_gt_conv = ineq_stats
print(f"[ineq] < candidates: {n_lt_cand:,} | converted: {n_lt_conv:,} (applied -{EPS})")
print(f"[ineq] > candidates: {n_gt_cand:,} | converted: {n_gt_conv:,} (applied +{EPS})")

# ----- STEP F: drop explicit admin/technical error rows from numeric pipeline (as before) -----
before = labs.height
labs = labs.filter(~pl.col("IS_ERROR_TEXT"))
print(f"Dropped {before - labs.height:,} rows marked as admin/technical error text.")

# ----- STEP G: drop any remaining TEXT tokens that are rare (<5% of that ITEMID’s rows) -----
# Define a normalized token for VALUE text (uppercase, strip punctuation/extra spaces).
# only consider rows flagged IS_TEXT_VALUE; inequality rows were excluded earlier.
def _norm_token(s: str) -> str | None:
    if s is None:
        return None
    t = re.sub(r"[\s]+", " ", re.sub(r"[^\w\s+<>=.\-]", " ", s.strip().upper()))  # keep +,-,.,<,>,=
    return t if t else None

labs = labs.with_columns(
    pl.when(pl.col("IS_TEXT_VALUE"))
      .then(pl.col("VALUE").map_elements(_norm_token, return_dtype=pl.Utf8))
      .otherwise(None)
      .alias("_TEXT_TOKEN")
)

# Compute per-ITEMID totals and token counts
totals_by_item = labs.group_by("ITEMID").agg(pl.len().alias("_n_item_rows"))

token_counts = (
    labs.filter(pl.col("_TEXT_TOKEN").is_not_null())
        .group_by(["ITEMID", "_TEXT_TOKEN"])
        .agg(pl.len().alias("_n_token_rows"))
        .join(totals_by_item, on="ITEMID", how="left")
        .with_columns((pl.col("_n_token_rows") / pl.col("_n_item_rows")).alias("_token_share"))
)

# Identify rare tokens (<5% of that ITEMID’s rows)
RARE_THRESH = 0.05
rare_tokens = token_counts.filter(pl.col("_token_share") < RARE_THRESH).select(["ITEMID","_TEXT_TOKEN"])

# Drop rows whose text token is rare
before = labs.height
labs = labs.join(rare_tokens, on=["ITEMID","_TEXT_TOKEN"], how="anti")
dropped_rare = before - labs.height
print(f"Dropped {dropped_rare:,} rows with rare text tokens (<5% per ITEMID).")

# ----- STEP G2: drop residual non-converted VALUE rows if they are <10% for that ITEMID -----
# "Uncaught" = VALUE present but no numeric available (not an error row, and both VALUENUM and VAL_FINAL are null)
UNCOV_THRESH = 0.10  # 10%

labs = labs.with_columns(
    (
        pl.col("VALUE").is_not_null()
        & ~pl.col("IS_ERROR_TEXT")
        & pl.col("VALUENUM").is_null()
        & pl.col("VAL_FINAL").is_null()
    ).alias("_is_uncaught")
)

# Compute share of uncaught rows per ITEMID
uncaught_stats = (
    labs.group_by("ITEMID")
        .agg([
            pl.len().alias("_n_rows"),
            pl.col("_is_uncaught").sum().alias("_n_uncaught")
        ])
        .with_columns((pl.col("_n_uncaught") / pl.col("_n_rows")).alias("_uncaught_share"))
)

# Apply the drop only for ITEMIDs where the uncaught share is below 10%
itemids_ok_to_drop = (
    uncaught_stats
        .filter(pl.col("_uncaught_share") < UNCOV_THRESH)
        .select("ITEMID")
        .to_series()
        .to_list()
)

before = labs.height
labs = labs.filter(
    ~(
        pl.col("ITEMID").is_in(itemids_ok_to_drop)
        & pl.col("_is_uncaught")
    )
)
dropped_uncaught = before - labs.height
print(f"Dropped {dropped_uncaught:,} residual non-converted rows (<{int(UNCOV_THRESH*100)}% per ITEMID).")

# cleanup helper columns
labs = labs.drop(["_is_uncaught"])

# ----- STEP H: round VAL_FINAL for consistency -----
labs = labs.with_columns(pl.col("VAL_FINAL").round(2))

# ----- STEP H.1: publish a single numeric & (optionally) clean the text -----
# 1) Single numeric for analysis/audit: prefer original VALUENUM else converted VAL_FINAL
labs = labs.with_columns(
    pl.coalesce([pl.col("VALUENUM"), pl.col("VAL_FINAL")]).round(2).alias("VALUENUM_CLEAN")
)

# 2) rewrite VALUE text for rows that were inequalities so audit won’t show "greater than ..."
labs = labs.with_columns(
    pl.when(pl.col("_has_lt") | pl.col("_has_gt"))
      .then(pl.col("VALUENUM_CLEAN").cast(pl.Utf8))  # replace phrase with the final numeric string
      .otherwise(pl.col("VALUE"))
      .alias("VALUE")
)

# 3) if downstream expects the column name VALUENUM, replace it in-place:
labs = labs.drop(["VALUENUM"]).rename({"VALUENUM_CLEAN": "VALUENUM"})


# ---- full-row dedupe (kept) ----
before = labs.height
labs = labs.unique(maintain_order=True)  # full-row dedupe
print(f"Removed {before - labs.height:,} exact duplicate rows.")

# ----- Final column selection & order -----
FINAL_COLS = [
    "SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME",
    "VALUENUM", "VALUE",
    "VALUEUOM", "LABEL", "CATEGORY", "FLUID",
    "FLAG_CLEAN", "IS_ABNORMAL",
]

# Warn if anything unexpected exists, then select only the final set
unexpected = [c for c in labs.columns if c not in FINAL_COLS]
if unexpected:
    print(f"[FINAL] Dropping {len(unexpected)} helper/raw columns: {unexpected[:12]}{' ...' if len(unexpected)>12 else ''}")

labs = labs.select(FINAL_COLS)

# ---- Profiling summary ----
prof_before = profile_dataframe(labs_before, table_name="LABEVENTS_sampled")
prof_after  = profile_dataframe(labs,        table_name="LABEVENTS_cleaned")
comparison  = compare_profiles(prof_before, prof_after)

summary_path = os.path.join(OUTPUT_DIR, "LABEVENTS_cleaning_summary.csv")
comparison.write_csv(summary_path)
print(f"[REPORT] Wrote before/after summary → {summary_path}")

# ---- Save cleaned file ----
out_path = os.path.join(OUTPUT_DIR, "LABEVENTS_cleaned.parquet")
labs.write_parquet(out_path, compression="zstd")
print(f"Saved cleaned LABEVENTS → {out_path}")
print("Done.")
