# 05_05_attach_rx.py — simplified quantitative prescription features
# This file attaches prescription features to the main spine table.
#In medicine, Rx is a common abbreviation for prescriptions.

import polars as pl
import os, pathlib

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

RX_PATH   = os.path.join(CLEAN_DIR, "PRESCRIPTIONS_cleaned.parquet")
SPINE_IN  = os.path.join(FEAT_DIR, "05_04_attach_proc.parquet")
OUT_PARQUET = os.path.join(FEAT_DIR, "05_05_attach_rx.parquet")
OUT_SUMMARY = os.path.join(FEAT_DIR, "05_05_attach_rx_summary.csv")

# ---------- Load ----------
rx = pl.read_parquet(RX_PATH)
spine = pl.read_parquet(SPINE_IN)

print(f"[LOAD] PRESCRIPTIONS_cleaned: {rx.height:,} rows, {rx.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")

# ---------- Ensure proper datatypes ----------
rx = rx.with_columns([
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("DRUG").cast(pl.Utf8).str.to_uppercase(),
    pl.col("STARTDATE").cast(pl.Datetime),
    pl.col("ENDDATE").cast(pl.Datetime),
])

# ---------- Duration (in days) ----------
rx = rx.with_columns([
    (pl.col("ENDDATE") - pl.col("STARTDATE")).dt.total_days().alias("RX_DURATION_DAYS")
])

# ---------- Detect overlaps within same admission ----------
# Overlap: if any pair of START/END periods overlap
rx_sorted = rx.sort(["HADM_ID", "STARTDATE"])
rx_overlap = (
    rx_sorted.with_columns([
        (pl.col("STARTDATE").shift(-1) < pl.col("ENDDATE")).alias("OVERLAP_NEXT")
    ])
    .group_by("HADM_ID")
    .agg(pl.col("OVERLAP_NEXT").any().cast(pl.Int8).alias("RX_HAS_OVERLAP"))
)

# ---------- Aggregate to admission level ----------
rx_agg = (
    rx.group_by("HADM_ID")
      .agg([
          pl.len().alias("RX_N_ROWS"),
          pl.col("DRUG").n_unique().alias("RX_N_UNIQUE_DRUGS"),
          pl.col("RX_DURATION_DAYS").mean().round(2).alias("RX_MEAN_DURATION_DAYS"),
          pl.col("RX_DURATION_DAYS").sum().round(2).alias("RX_TOTAL_DAYS"),
      ])
      .join(rx_overlap, on="HADM_ID", how="left")
      .with_columns([
          (pl.col("RX_N_UNIQUE_DRUGS") > 5).cast(pl.Int8).alias("RX_POLYPHARMACY"),
      ])
)

print(f"[AGG] Aggregated RX features to admission level: {rx_agg.height:,} rows")

# ---------- Join into spine ----------
spine1 = spine.join(rx_agg, on="HADM_ID", how="left")

# ---------- Fill defaults ----------
spine1 = spine1.with_columns([
    pl.col("RX_N_ROWS").fill_null(0),
    pl.col("RX_N_UNIQUE_DRUGS").fill_null(0),
    pl.col("RX_MEAN_DURATION_DAYS").fill_null(0.0),
    pl.col("RX_TOTAL_DAYS").fill_null(0.0),
    pl.col("RX_POLYPHARMACY").fill_null(0).cast(pl.Int8),
    pl.col("RX_HAS_OVERLAP").fill_null(0).cast(pl.Int8),
])

# ---------- Derived ratios ----------
spine1 = spine1.with_columns([
    pl.when(pl.col("ADM_LOS_DAYS") > 0)
      .then((pl.col("RX_N_UNIQUE_DRUGS") / pl.col("ADM_LOS_DAYS")).round(3))
      .otherwise(0)
      .alias("RX_DRUGS_PER_DAY"),
])

# ---------- One-row summary ----------
summary = spine1.select([
    pl.len().alias("n_rows"),
    pl.col("RX_N_ROWS").mean().round(2).alias("mean_rx_rows"),
    pl.col("RX_N_UNIQUE_DRUGS").mean().round(2).alias("mean_unique_drugs"),
    pl.col("RX_MEAN_DURATION_DAYS").mean().round(2).alias("mean_duration_days"),
    pl.col("RX_DRUGS_PER_DAY").mean().round(3).alias("mean_drugs_per_day"),
    pl.col("RX_POLYPHARMACY").mean().round(3).alias("pct_polypharmacy"),
    pl.col("RX_HAS_OVERLAP").mean().round(3).alias("pct_overlap"),
])
summary.write_csv(OUT_SUMMARY)
print(f"[REPORT] Summary → {OUT_SUMMARY}")

# ---------- Save ----------
spine1.write_parquet(OUT_PARQUET, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PARQUET}")
print("[DONE]")
