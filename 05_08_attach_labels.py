# 05_08_attach_labels.py 
# join 30-day readmission label (UNPLANNED ONLY → LBL_READMIT_30D)
import polars as pl
import os, pathlib

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

ADM_PATH  = os.path.join(CLEAN_DIR, "ADMISSIONS_cleaned.parquet")
SPINE_IN  = os.path.join(FEAT_DIR, "05_07_attach_labs.parquet")   # most recent spine
OUT_PQ    = os.path.join(FEAT_DIR, "05_08_attach_labels.parquet")
OUT_CSV   = os.path.join(FEAT_DIR, "05_08_attach_labels_summary.csv")

# ---------- Load ----------
ad    = pl.read_parquet(ADM_PATH)
spine = pl.read_parquet(SPINE_IN)

print(f"[LOAD] ADMISSIONS_cleaned: {ad.height:,} rows, {ad.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")

need = {"HADM_ID","SUBJECT_ID","ADMITTIME","DISCHTIME"}
missing = need - set(ad.columns)
if missing:
    raise ValueError(f"[ERROR] ADMISSIONS missing columns: {missing}")

has_death = "DEATHTIME" in ad.columns
has_type  = "ADMISSION_TYPE" in ad.columns

# ---------- Types ----------
ad = ad.with_columns([
    pl.col("HADM_ID").cast(pl.Int64),
    pl.col("SUBJECT_ID").cast(pl.Int64),
    pl.col("ADMITTIME").cast(pl.Datetime),
    pl.col("DISCHTIME").cast(pl.Datetime),
    *( [pl.col("DEATHTIME").cast(pl.Datetime)] if has_death else [] ),
    *( [pl.col("ADMISSION_TYPE").cast(pl.Utf8)] if has_type else [] ),
])

# ---------- Next admission per subject ----------
ad_sorted = ad.sort(["SUBJECT_ID","ADMITTIME"])
ad_next = ad_sorted.with_columns([
    pl.col("HADM_ID").shift(-1).over("SUBJECT_ID").alias("NEXT_HADM_ID"),
    pl.col("ADMITTIME").shift(-1).over("SUBJECT_ID").alias("NEXT_ADMITTIME"),
    *( [pl.col("ADMISSION_TYPE").shift(-1).over("SUBJECT_ID").alias("NEXT_ADMISSION_TYPE")] if has_type else [] ),
])

ad_next = ad_next.with_columns(
    (pl.col("NEXT_ADMITTIME") - pl.col("DISCHTIME")).dt.total_days().alias("LBL_DAYS_TO_NEXT_ADMIT")
)

# ---------- Eligibility ----------
elig_expr = pl.col("DISCHTIME").is_not_null()
if has_death:
    elig_expr = elig_expr & pl.col("DEATHTIME").is_null()
ad_next = ad_next.with_columns(elig_expr.cast(pl.Int8).alias("LBL_READMIT_ELIGIBLE"))

# ---------- 30-day UNPLANNED label (standardized as LBL_READMIT_30D) ----------
any30 = (
    (pl.col("LBL_DAYS_TO_NEXT_ADMIT").is_not_null()) &
    (pl.col("LBL_DAYS_TO_NEXT_ADMIT") >= 0) &
    (pl.col("LBL_DAYS_TO_NEXT_ADMIT") <= 30)
)

if has_type:
    # treat anything starting with "ELECT" as elective; robust to variations like "Elective Admission"
    next_type_up = pl.col("NEXT_ADMISSION_TYPE").str.to_uppercase()
    is_elective  = next_type_up.str.starts_with("ELECT")
    unplanned30  = any30 & (~is_elective)
else:
    unplanned30  = any30

ad_next = ad_next.with_columns(
    (unplanned30 & (pl.col("LBL_READMIT_ELIGIBLE") == 1)).cast(pl.Int8).alias("LBL_READMIT_30D")
)

# Keep only join keys + final labels
labels = ad_next.select([
    "HADM_ID",
    "LBL_READMIT_ELIGIBLE",
    "LBL_DAYS_TO_NEXT_ADMIT",
    "LBL_READMIT_30D",
])

# ---------- Join to spine ----------
n0 = spine.select(pl.col("HADM_ID").n_unique()).item()
spine1 = spine.join(labels, on="HADM_ID", how="left").with_columns([
    pl.col("LBL_READMIT_ELIGIBLE").fill_null(0).cast(pl.Int8),
    pl.col("LBL_READMIT_30D").fill_null(0).cast(pl.Int8),
    # LBL_DAYS_TO_NEXT_ADMIT left as null if unknown
])
n1 = spine1.select(pl.col("HADM_ID").n_unique()).item()
assert n0 == n1, f"[ERROR] HADM_ID count changed after join: before={n0}, after={n1}"

# ---------- Output ----------
n_total = spine1.height
n_readmit = spine1.filter(pl.col("LBL_READMIT_30D") == 1).height
pct_readmit = n_readmit / n_total * 100
print(f"[STATS] {n_readmit:,} of {n_total:,} unplanned admissions were readmitted within 30 days ({pct_readmit:.2f}%)")

# ---------- Summary ----------
def _safe_item(df, expr: pl.Expr):
    out = df.select(expr)
    return None if out.is_empty() else out.item()

summary = pl.DataFrame([{
    "n_rows": spine1.height,
    "pct_eligible": _safe_item(spine1, pl.col("LBL_READMIT_ELIGIBLE").mean().round(4)),
    "pct_readmit30_unplanned": _safe_item(spine1, pl.col("LBL_READMIT_30D").mean().round(4)),
    "mean_days_to_next_among_unplanned": _safe_item(
        spine1.filter(pl.col("LBL_READMIT_30D") == 1),
        pl.col("LBL_DAYS_TO_NEXT_ADMIT").mean().round(2)
    ),
}])

summary.write_csv(OUT_CSV)
print(f"[REPORT] Summary → {OUT_CSV}")

# ---------- Save ----------
spine1.write_parquet(OUT_PQ, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PQ}")
print("[DONE]")
