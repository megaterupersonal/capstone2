# 05_01_attach_icu.py
# This file attaches ICU stay features to the main spine table.

import polars as pl
import os, pathlib

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"

ICU_PATH  = os.path.join(CLEAN_DIR, "ICUSTAYS_cleaned.parquet")
ADM_PATH  = os.path.join(CLEAN_DIR, "ADMISSIONS_cleaned.parquet")
SPINE_IN  = os.path.join(FEAT_DIR, "05_00_makespine.parquet")
SPINE_OUT = os.path.join(FEAT_DIR, "05_01_attach_icu.parquet")
SUM_CSV   = os.path.join(FEAT_DIR, "ICU_attach_summary.csv")

pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
icu = pl.read_parquet(ICU_PATH)
spine = pl.read_parquet(SPINE_IN)
ad = pl.read_parquet(ADM_PATH).select(["HADM_ID", "ADMITTIME"])

print(f"[LOAD] ICUSTAYS_cleaned: {icu.height:,} rows, {icu.width} cols")
print(f"[LOAD] SPINE: {spine.height:,} rows, {spine.width} cols")
print(f"[LOAD] ADMISSIONS_cleaned (for ADMITTIME): {ad.height:,} rows")

# ---------- Ensure types / minimal fix-ups ----------
icu = icu.with_columns([
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("INTIME").cast(pl.Datetime),
    pl.col("OUTTIME").cast(pl.Datetime),
])

# LOS source: prefer provided LOS (days) if present, otherwise compute
has_los_col = "LOS" in icu.columns
icu = icu.with_columns([
    pl.when(pl.lit(has_los_col))
      .then(pl.col("LOS").cast(pl.Float64))
      .otherwise((pl.col("OUTTIME") - pl.col("INTIME")).dt.total_hours() / 24.0)
      .alias("ICU_LOS_DAYS")
]).with_columns(pl.col("ICU_LOS_DAYS").round(2))

# Drop entries without HADM_ID or INTIME (cannot aggregate)
before = icu.height
icu = icu.filter(
    pl.col("HADM_ID").is_not_null() &
    pl.col("INTIME").is_not_null()
)
print(f"[CLEAN] Dropped {before - icu.height:,} ICU rows with null HADM_ID/INTIME.")

# ---------- Derive features within ICUSTAYS ----------
has_first_cu = "FIRST_CAREUNIT" in icu.columns
has_last_cu  = "LAST_CAREUNIT" in icu.columns

agg_exprs = [
    pl.len().alias("ICU_N_STAYS"),
    (pl.len() > 1).cast(pl.Int8).alias("ICU_READMIT_WITHIN_ADM"),
    pl.col("ICU_LOS_DAYS").sum().round(2).alias("ICU_TOTAL_LOS_DAYS"),
    pl.col("ICU_LOS_DAYS").max().round(2).alias("ICU_MAX_LOS_DAYS"),
    pl.col("INTIME").min().alias("ICU_FIRST_INTIME"),
    pl.col("OUTTIME").max().alias("ICU_LAST_OUTTIME"),
]
if has_first_cu:
    agg_exprs.append(pl.col("FIRST_CAREUNIT").mode().alias("ICU_FIRST_CAREUNIT_MODE"))
if has_last_cu:
    agg_exprs.append(pl.col("LAST_CAREUNIT").mode().alias("ICU_LAST_CAREUNIT_MODE"))

icu_agg = (
    icu.group_by("HADM_ID")
       .agg(agg_exprs)
       .with_columns([
           (pl.col("ICU_N_STAYS") > 0).cast(pl.Int8).alias("ICU_ANY"),
       ])
)

print(f"[AGG] Aggregated ICU features to admissions level: {icu_agg.height:,} rows")

# ---------- Join ICU → spine ----------
# (Left join keeps all admissions in the spine; fills 0/NULLs for non-ICU cases)
spine1 = spine.join(icu_agg, on="HADM_ID", how="left")

# Fill sensible defaults for non-ICU admissions
spine1 = spine1.with_columns([
    pl.col("ICU_ANY").fill_null(0).cast(pl.Int8),
    pl.col("ICU_N_STAYS").fill_null(0),
    pl.col("ICU_READMIT_WITHIN_ADM").fill_null(0).cast(pl.Int8),
    pl.col("ICU_TOTAL_LOS_DAYS").fill_null(0.0).round(2),
    pl.col("ICU_MAX_LOS_DAYS").fill_null(0.0).round(2),
])

# ---------- Cross-table feature: time to first ICU from hospital admit ----------
# Need ADMITTIME (not stored in spine). Bring it temporarily from ADMISSIONS.
spine1 = (
    spine1.join(ad.with_columns(pl.col("ADMITTIME").cast(pl.Datetime)),
                on="HADM_ID", how="left")
           .with_columns([
               pl.when(pl.col("ICU_FIRST_INTIME").is_not_null() & pl.col("ADMITTIME").is_not_null())
                 .then((pl.col("ICU_FIRST_INTIME") - pl.col("ADMITTIME")).dt.total_hours())
                 .otherwise(None)
                 .alias("ICU_HOURS_TO_FIRST_FROM_ADMIT")
           ])
           .drop(["ADMITTIME"])  # drop temp column
)

# ---------- Quick sanity / summary ----------
summary = (
    spine1.select([
        pl.len().alias("n_rows"),
        pl.col("ICU_ANY").mean().round(3).alias("pct_with_any_icu"),
        pl.col("ICU_N_STAYS").mean().round(2).alias("mean_icu_stays"),
        pl.col("ICU_TOTAL_LOS_DAYS").mean().round(2).alias("mean_icu_total_los_days"),
        pl.col("ICU_HOURS_TO_FIRST_FROM_ADMIT").drop_nulls().mean().round(2).alias("mean_hours_to_first_icu"),
    ])
)
print(summary)

# Save a CSV summary for the appendix
summary.with_columns([
    pl.lit("ICU attach summary").alias("note")
]).write_csv(SUM_CSV)
print(f"[REPORT] Wrote summary → {SUM_CSV}")

# ---------- Save updated spine ----------
spine1.write_parquet(SPINE_OUT, compression="zstd")
print(f"[SAVE] Wrote spine + ICU features → {SPINE_OUT}")
print("[DONE]")
