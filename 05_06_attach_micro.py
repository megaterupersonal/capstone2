# 05_06_attach_micro.py 
# join microbiology infection flags
import polars as pl
import os, pathlib

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

# If your cleaned file has a different name, adjust here:
MICRO_PATH = os.path.join(CLEAN_DIR, "MICROBIOLOGY_infection_by_hadm.parquet")
SPINE_IN   = os.path.join(FEAT_DIR, "05_05_attach_rx.parquet")

OUT_PARQUET = os.path.join(FEAT_DIR, "05_06_attach_micro.parquet")
OUT_SUMMARY = os.path.join(FEAT_DIR, "05_06_attach_micro_summary.csv")

# ---------- Load ----------
micro = pl.read_parquet(MICRO_PATH)
spine = pl.read_parquet(SPINE_IN)

print(f"[LOAD] MICROBIOLOGYEVENTS_cleaned: {micro.height:,} rows, {micro.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")

# ---------- Make sure required columns exist ----------
need_cols = {"HADM_ID", "HAS_INFECTION"}
missing = need_cols - set(micro.columns)
if missing:
    raise ValueError(f"[ERROR] Microbiology file missing columns: {missing}")

# ---------- Join into spine (left) & fill defaults ----------
have_counts = {"MICRO_SRC_N_ROWS", "MICRO_SRC_N_POS"}.issubset(set(micro.columns))

if have_counts:
    micro_agg = micro.select([
        "HADM_ID",
        pl.col("MICRO_SRC_N_ROWS").alias("MICRO_N_ROWS"),
        pl.col("MICRO_SRC_N_POS").alias("MICRO_N_POSITIVE"),
        pl.col("HAS_INFECTION").cast(pl.Int8).alias("MICRO_HAS_INFECTION"),
    ])
else:
    # Back-compat: if the parquet only has HAS_INFECTION (one row/HADM_ID), emulate old behavior
    micro_agg = (
        micro.group_by("HADM_ID")
             .agg([
                 pl.len().alias("MICRO_N_ROWS"),
                 pl.col("HAS_INFECTION").cast(pl.Int8).sum().alias("MICRO_N_POSITIVE"),
                 pl.col("HAS_INFECTION").cast(pl.Int8).max().alias("MICRO_HAS_INFECTION"),
             ])
    )

spine1 = (
    spine.join(micro_agg, on="HADM_ID", how="left")
         .with_columns([
             pl.col("MICRO_N_ROWS").fill_null(0),
             pl.col("MICRO_N_POSITIVE").fill_null(0),
             pl.col("MICRO_HAS_INFECTION").fill_null(0).cast(pl.Int8),
         ])
)

# ---------- One-row summary ----------
summary = spine1.select([
    pl.len().alias("n_rows"),
    pl.col("MICRO_HAS_INFECTION").mean().round(4).alias("pct_any_infection"),
    pl.col("MICRO_N_ROWS").mean().round(2).alias("mean_micro_rows"),
    pl.col("MICRO_N_POSITIVE").mean().round(2).alias("mean_micro_positive_rows"),
])
summary.write_csv(OUT_SUMMARY)
print(f"[REPORT] Summary → {OUT_SUMMARY}")

# ---------- Save ----------
spine1.write_parquet(OUT_PARQUET, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PARQUET}")
print("[DONE]")
