# 05_02_attach_transfers.py
# This file attaches TRANSFERS stay features to the main spine table.

import polars as pl
import os, pathlib
import re

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"

XFER_PATH = os.path.join(CLEAN_DIR, "TRANSFERS_cleaned.parquet")
ADM_PATH  = os.path.join(CLEAN_DIR, "ADMISSIONS_cleaned.parquet")


SPINE_IN = os.path.join(FEAT_DIR, "05_01_attach_icu.parquet")
SPINE_OUT = os.path.join(FEAT_DIR, "05_02_attach_transfers.parquet")
SUM_CSV   = os.path.join(FEAT_DIR, "05_02_attach_transfers_summary.csv")

pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Load ----------
xfer = pl.read_parquet(XFER_PATH)
spine = pl.read_parquet(SPINE_IN)
ad = pl.read_parquet(ADM_PATH).select(["HADM_ID", "ADMITTIME", "DISCHTIME"])

print(f"[LOAD] TRANSFERS_cleaned: {xfer.height:,} rows, {xfer.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")
print(f"[LOAD] ADMISSIONS_cleaned (ADMIT/DISCH): {ad.height:,} rows")

# ---------- Ensure types / minimal fix-ups ----------
must_have = ["HADM_ID", "INTIME", "OUTTIME"]
for c in must_have:
    if c not in xfer.columns:
        raise ValueError(f"TRANSFERS is missing required column: {c}")

xfer = xfer.with_columns([
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("INTIME").cast(pl.Datetime),
    pl.col("OUTTIME").cast(pl.Datetime),
])

has_curr = "CURR_CAREUNIT" in xfer.columns
has_prev = "PREV_CAREUNIT" in xfer.columns
has_event = "EVENTTYPE" in xfer.columns
has_source = "DBSOURCE" in xfer.columns

# Define ICU careunits for tagging (case-insensitive). This indicates a transfer was done based on ICU stay.
icu_units = ["MICU", "SICU", "CCU", "CSRU", "TSICU", "NICU", "NWARD", "Neuro Stepdown"]
# Match via regex union; include word boundaries where possible
icu_regex = "(?i)" + "|".join([re.escape(u) for u in icu_units])

# Tag ICU rows if we have a careunit column
if has_curr:
    xfer = xfer.with_columns(
        pl.col("CURR_CAREUNIT").cast(pl.Utf8).str.contains(icu_regex, literal=False)
        .fill_null(False).cast(pl.Int8).alias("XFER_IS_ICU_UNIT")
    )
else:
    xfer = xfer.with_columns(pl.lit(None, dtype=pl.Int8).alias("XFER_IS_ICU_UNIT"))

# Define transfer vs admit/discharge if EVENTTYPE exists
if has_event:
    xfer = xfer.with_columns([
        (pl.col("EVENTTYPE").cast(pl.Utf8).str.to_lowercase() == "transfer")
            .cast(pl.Int8).alias("XFER_IS_TRANSFER")
    ])
else:
    # If missing, approximate: count every row as a "stop"; transfers will be stops-1 later
    xfer = xfer.with_columns(pl.lit(None, dtype=pl.Int8).alias("XFER_IS_TRANSFER"))

# ---------- Derive features within TRANSFERS (per HADM_ID) ----------
agg_exprs = [
    pl.len().alias("TR_N_STOPS"),
    pl.col("INTIME").min().alias("TR_FIRST_INTIME"),
    pl.col("OUTTIME").max().alias("TR_LAST_OUTTIME"),
]

if has_event:
    # sum the 0/1 "transfer" flags; coalesce to 0 after aggregation
    agg_exprs.append(pl.col("XFER_IS_TRANSFER").sum().alias("TR_N_TRANSFERS"))
else:
    # fallback: estimate transfers as max(stops-1, 0)
    agg_exprs.append((pl.len() - 1).clip_lower(0).alias("TR_N_TRANSFERS"))

if has_curr:
    agg_exprs.extend([
        pl.col("CURR_CAREUNIT").n_unique().alias("TR_N_UNIQUE_UNITS"),
        # count ICU stops (0/1), sum ignores nulls; coalesce later
        pl.col("XFER_IS_ICU_UNIT").sum().alias("TR_N_ICU_STOPS"),
        (pl.col("XFER_IS_ICU_UNIT").max().fill_null(0)).cast(pl.Int8).alias("TR_ANY_ICU_UNIT"),
    ])
else:
    agg_exprs.extend([
        pl.lit(None).alias("TR_N_UNIQUE_UNITS"),
        pl.lit(None).alias("TR_N_ICU_STOPS"),
        pl.lit(0).cast(pl.Int8).alias("TR_ANY_ICU_UNIT"),
    ])

# first ICU intime if any
if has_curr:
    xfer = xfer.with_columns(
        pl.when(pl.col("XFER_IS_ICU_UNIT") == 1)
          .then(pl.col("INTIME"))
          .otherwise(None)
          .alias("INTIME_IF_ICU")
    )

icu_first_expr = (
    pl.col("INTIME_IF_ICU").min().alias("TR_FIRST_ICU_INTIME")
    if "INTIME_IF_ICU" in xfer.columns else pl.lit(None).alias("TR_FIRST_ICU_INTIME")
)

xfer_agg = (
    xfer.group_by("HADM_ID").agg(agg_exprs + [icu_first_expr])
      .with_columns([
          pl.col("TR_N_TRANSFERS").fill_null(0).cast(pl.Int64),
          pl.col("TR_N_ICU_STOPS").fill_null(0).cast(pl.Int64),
      ])
)


print(f"[AGG] Aggregated transfer features to admissions level: {xfer_agg.height:,} rows")

# ---------- Join TRANSFERS → spine (left) ----------
spine1 = spine.join(xfer_agg, on="HADM_ID", how="left")

# Fill defaults for admissions with no transfers recorded
spine1 = spine1.with_columns([
    pl.col("TR_N_STOPS").fill_null(0),
    pl.col("TR_N_TRANSFERS").fill_null(0),
    pl.col("TR_N_UNIQUE_UNITS").fill_null(0),
    pl.col("TR_N_ICU_STOPS").fill_null(0),
    pl.col("TR_ANY_ICU_UNIT").fill_null(0).cast(pl.Int8),
])

# ---------- Cross-table features ----------
# Bring ADMIT/DISCH to compute timing-based features; drop afterwards
spine1 = (
    spine1.join(ad.with_columns([
                pl.col("ADMITTIME").cast(pl.Datetime),
                pl.col("DISCHTIME").cast(pl.Datetime)]),
                on="HADM_ID", how="left")
          .with_columns([
              # Hours from hospital admit to first transfer stop
              pl.when(pl.col("TR_FIRST_INTIME").is_not_null() & pl.col("ADMITTIME").is_not_null())
                .then((pl.col("TR_FIRST_INTIME") - pl.col("ADMITTIME")).dt.total_hours())
                .otherwise(None)
                .alias("TR_HOURS_TO_FIRST_FROM_ADMIT"),

              # Hours from last transfer stop to hospital discharge
              pl.when(pl.col("TR_LAST_OUTTIME").is_not_null() & pl.col("DISCHTIME").is_not_null())
                .then((pl.col("DISCHTIME") - pl.col("TR_LAST_OUTTIME")).dt.total_hours())
                .otherwise(None)
                .alias("TR_HOURS_LAST_TO_DISCH"),

              # Transfer intensity normalized by LOS (bed moves per hospital day)
              pl.when(pl.col("ADM_LOS_DAYS").is_not_null() & (pl.col("ADM_LOS_DAYS") > 0))
                .then((pl.col("TR_N_TRANSFERS") / pl.col("ADM_LOS_DAYS")).round(3))
                .otherwise(None)
                .alias("TR_TRANSFERS_PER_DAY"),
          ])
          .drop(["ADMITTIME", "DISCHTIME"])  # temp columns removed
)

# Clip any negative timing artifacts (rare due to date-shift)
for col in ["TR_HOURS_TO_FIRST_FROM_ADMIT", "TR_HOURS_LAST_TO_DISCH"]:
    if col in spine1.columns:
        spine1 = spine1.with_columns(
            pl.when(pl.col(col) < 0).then(0).otherwise(pl.col(col)).alias(col)
        )

# ---------- Sanity check: left join kept all admissions ----------
n0 = spine.select(pl.col("HADM_ID").n_unique()).item()
n1 = spine1.select(pl.col("HADM_ID").n_unique()).item()
assert n0 == n1, f"Left-join changed row count: before={n0}, after={n1}"

# ---------- Quick summary ----------
summary = (
    spine1.select([
        pl.len().alias("n_rows"),
        pl.col("TR_N_TRANSFERS").mean().round(3).alias("mean_transfers"),
        pl.col("TR_TRANSFERS_PER_DAY").drop_nulls().mean().round(3).alias("mean_transfers_per_day"),
        pl.col("TR_ANY_ICU_UNIT").mean().round(3).alias("pct_any_icu_unit_in_transfers"),
        pl.col("TR_N_UNIQUE_UNITS").mean().round(2).alias("mean_unique_units"),
        pl.col("TR_HOURS_TO_FIRST_FROM_ADMIT").drop_nulls().median().round(2).alias("median_hours_to_first_transfer"),
        pl.col("TR_HOURS_LAST_TO_DISCH").drop_nulls().median().round(2).alias("median_hours_last_transfer_to_discharge"),
    ])
)
print(summary)

summary.with_columns(pl.lit("TRANSFERS attach summary").alias("note")).write_csv(SUM_CSV)
print(f"[REPORT] Wrote summary → {SUM_CSV}")

# ---------- Save updated spine ----------
spine1.write_parquet(SPINE_OUT, compression="zstd")
print(f"[SAVE] Wrote → {SPINE_OUT}")
print("[DONE]")
