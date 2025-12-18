# 05_00_make_spine.py
# This file creates the main spine table by combining cleaned ADMISSIONS and PATIENTS data, deriving key features.

import polars as pl
import os, pathlib

# ---------- Paths ----------
PARQUET_DIR = r"H:\A\parquet files\cleaned"
OUT_DIR     = r"H:\A\parquet files\features"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

ADM_PATH = os.path.join(PARQUET_DIR, "ADMISSIONS_cleaned.parquet")
PAT_PATH = os.path.join(PARQUET_DIR, "PATIENTS_cleaned.parquet")
OUT_SPINE = os.path.join(OUT_DIR, "05_00_makespine.parquet")

# ---------- Load ----------
ad = pl.read_parquet(ADM_PATH)
pt = pl.read_parquet(PAT_PATH)

print(f"[LOAD] ADMISSIONS_cleaned: {ad.height:,} rows, {ad.width} cols")
print(f"[LOAD] PATIENTS_cleaned : {pt.height:,} rows, {pt.width} cols")

# ---------- Derive features: ADMISSIONS ----------
# Notes:
# - Keep HADM_ID/SUBJECT_ID as keys
# - Add temporal and simple admin flags
# --- build ad_feats (ADD ADMITTIME to the select) ---
ad_feats = (
    ad.with_columns([
        # --- temporal ---
        pl.col("ADMITTIME").alias("ADMITTIME"),
        pl.col("ADMITTIME").dt.year().alias("ADM_YEAR"),
        pl.col("ADMITTIME").dt.month().alias("ADM_MONTH"),
        pl.col("ADMITTIME").dt.weekday().alias("ADM_WEEKDAY"),

        # --- admission type flags ---
        (pl.col("ADMISSION_TYPE") == "ELECTIVE").cast(pl.Int8).alias("ADM_IS_ELECTIVE"),
        (pl.col("ADMISSION_TYPE") == "EMERGENCY").cast(pl.Int8).alias("ADM_IS_EMERGENCY"),
        (pl.col("ADMISSION_TYPE") == "URGENT").cast(pl.Int8).alias("ADM_IS_URGENT"),

        # --- discharge to facility (not home) ---
        (~pl.col("DISCHARGE_LOCATION").cast(pl.Utf8)
          .str.contains("(?i)HOME", literal=False)).cast(pl.Int8).alias("ADM_DISCH_TO_FACILITY"),

        # --- public insurance flag ---
        pl.when(pl.col("INSURANCE").cast(pl.Utf8).str.contains("(?i)Medicare|Medicaid", literal=False))
          .then(1).otherwise(0).cast(pl.Int8).alias("ADM_PUBLIC_INSURANCE"),

        # --- LOS ---
        pl.col("LOS_DAYS").cast(pl.Float64).round(2).alias("ADM_LOS_DAYS"),

        # --- carry categoricals (optional) ---
        pl.col("ADMISSION_TYPE").alias("ADM_TYPE"),
        pl.col("INSURANCE").alias("ADM_INSURANCE"),
        pl.col("DISCHARGE_LOCATION").alias("ADM_DISCHARGE_LOCATION"),

        # --- MARITAL: Married=1 else 0 ---
        (pl.col("MARITAL_STATUS").cast(pl.Utf8).str.to_uppercase().eq("MARRIED"))
            .cast(pl.Int8).alias("ADM_IS_CURRENTLYMARRIED"),

        # --- ETHNICITY grouped ---
        pl.when(pl.col("ETHNICITY").str.contains("(?i)WHITE", literal=False))
          .then(pl.lit("WHITE"))
          .when(pl.col("ETHNICITY").str.contains("(?i)BLACK", literal=False))
          .then(pl.lit("BLACK/AFRICAN AMERICAN"))
          .when(pl.col("ETHNICITY").str.contains("(?i)HISPANIC", literal=False))
          .then(pl.lit("HISPANIC/LATINO"))
          .when(pl.col("ETHNICITY").str.contains("(?i)UNKNOWN|UNSPECIFIED|NOT SPECIFIED", literal=False))
          .then(pl.lit("UNKNOWN"))
          .otherwise(pl.lit("OTHER"))
          .alias("ADM_ETHNICITY_GROUPED"),
    ])
    .select([
        "HADM_ID", "SUBJECT_ID",
        "ADMITTIME",
        "ADM_LOS_DAYS",
        "ADM_YEAR", "ADM_MONTH", "ADM_WEEKDAY",
        "ADM_IS_ELECTIVE", "ADM_IS_EMERGENCY", "ADM_IS_URGENT",
        "ADM_DISCH_TO_FACILITY", "ADM_PUBLIC_INSURANCE",
        "ADM_TYPE", "ADM_INSURANCE", "ADM_DISCHARGE_LOCATION",
        "ADM_IS_CURRENTLYMARRIED", "ADM_ETHNICITY_GROUPED",
    ])
)


# --- patient features ---
pt_feats = pt.select([
    "SUBJECT_ID",
    pl.col("GENDER").alias("PAT_GENDER"),
    pl.col("DOB").alias("PAT_DOB"),
])

# --- join (no row removal yet) ---
spine_base = ad_feats.join(pt_feats, on="SUBJECT_ID", how="left")

# sanity: BEFORE any filters, HADM_ID must be 1:1 with admissions
n_adm = ad.select(pl.col("HADM_ID").n_unique()).item()
n_spn0 = spine_base.select(pl.col("HADM_ID").n_unique()).item()
assert n_adm == n_spn0, f"[ERROR] HADM_ID count changed on join: adm={n_adm}, spine0={n_spn0}"

# --- compute age ---
spine_base = spine_base.with_columns([
    pl.col("ADMITTIME").cast(pl.Datetime),
    pl.col("PAT_DOB").cast(pl.Datetime),
    ((pl.col("ADMITTIME") - pl.col("PAT_DOB")).dt.total_days() / 365.25)
        .round(2).alias("PAT_AGE_AT_ADMIT")
]).drop(["ADMITTIME","PAT_DOB", "ADM_TYPE"])  # drop raws now

# --- handle MIMIC elderly masking (≥120 → HIPAA) ---
spine_base = spine_base.with_columns([
    # flag elderly rows per the masking rule
    (pl.col("PAT_AGE_AT_ADMIT") >= 120).cast(pl.Int8).alias("PAT_IS_ELDERLY"),
    # cap ages ≥120 to 89 for interpretability while keeping rows
    pl.when(pl.col("PAT_AGE_AT_ADMIT") >= 120)
      .then(pl.lit(89.0))
      .otherwise(pl.col("PAT_AGE_AT_ADMIT"))
      .alias("PAT_AGE_AT_ADMIT"),
])


# After capping, only drop neonates/infants (<1). Elderly are retained with PAT_IS_ELDERLY=1 and age=89. This is because of how MIMIC anonymizes elderly patients.
before_rows = spine_base.height
spine = spine_base.filter(pl.col("PAT_AGE_AT_ADMIT") >= 1)
after_rows = spine.height

n_removed   = before_rows - after_rows
n_null_age  = spine_base.filter(pl.col("PAT_AGE_AT_ADMIT").is_null()).height
n_lt1       = spine_base.filter(pl.col("PAT_AGE_AT_ADMIT") < 1).height
n_elderly   = spine_base.filter(pl.col("PAT_IS_ELDERLY") == 1).height  # kept, not removed
print(f"[FILTER] Removed {n_removed:,} rows (null_age={n_null_age:,}, age<1={n_lt1:,}). "
      f"Kept elderly (masked) rows: {n_elderly:,}.")

n_adm  = ad.select(pl.col("HADM_ID").n_unique()).item()
n_spn0 = spine_base.select(pl.col("HADM_ID").n_unique()).item()
assert n_adm == n_spn0, f"[ERROR] HADM_ID count changed on join: adm={n_adm}, spine0={n_spn0}"


# ---------- Save ----------
spine.write_parquet(OUT_SPINE, compression="zstd")
print(f"[SAVE] Wrote spine → {OUT_SPINE}")

# === Descriptive summary of spine ===
import numpy as np

def summarize_dataframe(df: pl.DataFrame, out_csv: str):
    """Generate simple descriptive stats for each column in df."""
    records = []
    for col, dtype in df.schema.items():
        dtype_str = str(dtype)
        s = df[col]
        rec = {
            "column": col,
            "dtype": dtype_str,
            "n_rows": s.len(),
            "n_nulls": s.null_count(),
            "n_unique": s.n_unique(),
            "mean": None,
            "median": None,
            "mode": None,
            "min": None,
            "max": None,
        }

        # numeric columns
        if any(t in dtype_str for t in ["Int", "Float"]):
            stats = df.select([
                pl.col(col).mean().alias("mean"),
                pl.col(col).median().alias("median"),
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ]).to_dict(as_series=False)
            for k, v in stats.items():
                rec[k] = round(v[0], 2) if v[0] is not None else None

        # datetime columns
        elif "Date" in dtype_str:
            stats = df.select([
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
            ]).to_dict(as_series=False)
            rec["min"], rec["max"] = stats["min"][0], stats["max"][0]

        # mode for all types (simple value_counts approach)
        try:
            vc = s.value_counts().sort("count", descending=True)
            if vc.height > 0:
                mode_col = vc.columns[0]  # get actual column name dynamically
                rec["mode"] = vc[mode_col][0]
        except Exception:
            rec["mode"] = None

        records.append(rec)

    out_df = pl.DataFrame(records)
    out_df.write_csv(out_csv)
    print(f"[REPORT] Descriptive summary saved → {out_csv}")

# === Generate and save ===
summary_path = os.path.join(OUT_DIR, "SPINE_descriptive_summary.csv")
summarize_dataframe(spine, summary_path)
