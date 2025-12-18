# 05_07_attach_labs.py 
# LABEVENTS > wide features (numeric + text) + join to spine
import polars as pl
import os, pathlib, re

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

LABS_PATH  = os.path.join(CLEAN_DIR, "LABEVENTS_cleaned.parquet")
SPINE_IN   = os.path.join(FEAT_DIR, "05_06_attach_micro.parquet")   
OUT_PQ     = os.path.join(FEAT_DIR, "05_07_attach_labs.parquet")
OUT_CSV    = os.path.join(FEAT_DIR, "05_07_attach_labs_summary.csv")

# ---------- Load ----------
labs  = pl.read_parquet(LABS_PATH)
spine = pl.read_parquet(SPINE_IN)

print(f"[LOAD] LABEVENTS_cleaned: {labs.height:,} rows, {labs.width} cols")
print(f"[LOAD] Spine in: {spine.height:,} rows, {spine.width} cols  ← {os.path.basename(SPINE_IN)}")

# Expect at minimum:
need = {"HADM_ID","CHARTTIME","VALUE","VALUENUM","VALUEUOM","LABEL","FLUID","IS_ABNORMAL"}
missing = need - set(labs.columns)
if missing:
    raise ValueError(f"[ERROR] LABEVENTS missing columns: {missing}")

# ---------- 3) Build test identifiers ----------
# TEST_KEY = FLUID + "|" + LABEL + ("|" + VALUEUOM if present)
labs = labs.with_columns([
    pl.col("FLUID").cast(pl.Utf8).fill_null("UNKFLUID").str.strip_chars(),
    pl.col("LABEL").cast(pl.Utf8).fill_null("UNKLABEL").str.strip_chars(),
    pl.col("VALUEUOM").cast(pl.Utf8).fill_null(""),
]).with_columns([
    pl.when(pl.col("VALUEUOM") == "")
      .then(pl.concat_str([pl.col("FLUID"), pl.col("LABEL")], separator="|"))
      .otherwise(pl.concat_str([pl.col("FLUID"), pl.col("LABEL"), pl.col("VALUEUOM")], separator="|"))
      .alias("TEST_KEY_RAW")
])

# safe column names for pivot
labs = labs.with_columns(
    pl.col("TEST_KEY_RAW")
      .str.replace_all(r"[^A-Za-z0-9]+", "_", literal=False)
      .str.strip_chars("_")
      .str.to_uppercase()
      .alias("TEST_KEY")
)

# ---------- 4) Collapse repeated measures: keep LATEST per (HADM_ID, TEST_KEY) ----------
labs = labs.with_columns([
    pl.col("CHARTTIME").cast(pl.Datetime),
    pl.col("VALUENUM").cast(pl.Float64),
    pl.col("IS_ABNORMAL").cast(pl.Int8).fill_null(0),
])

latest = (
    labs
    .sort("CHARTTIME")
    .group_by(["HADM_ID","TEST_KEY"])
    .agg([
        pl.col("VALUENUM").last().alias("VAL_NUM"),
        pl.col("VALUE").last().cast(pl.Utf8).alias("VAL_TEXT"),
        pl.col("IS_ABNORMAL").max().alias("ABN_FLAG"),
        pl.col("VALUEUOM").last().cast(pl.Utf8).alias("VALUEUOM"),
        pl.col("LABEL").last().cast(pl.Utf8).alias("LABEL"),
        pl.col("FLUID").last().cast(pl.Utf8).alias("FLUID"),
    ])
)

# ---------- split numeric vs textual ----------
num_latest  = latest.filter(pl.col("VAL_NUM").is_not_null()).with_columns(pl.col("VAL_NUM").round(2))
text_latest = latest.filter(pl.col("VAL_NUM").is_null() & pl.col("VAL_TEXT").is_not_null())

# ---------- 5) Pivot long → wide ----------
num_wide = (
    num_latest
    .pivot(index="HADM_ID", on="TEST_KEY", values="VAL_NUM", aggregate_function="first")
)
# Prefix numeric labs as LAB_*
num_ren = {c: ("LAB_" + c) for c in num_wide.columns if c != "HADM_ID"}
num_wide = num_wide.rename(num_ren)

abn_wide = (
    latest
    .pivot(index="HADM_ID", on="TEST_KEY", values="ABN_FLAG", aggregate_function="max")
)
abn_ren = {c: ("LAB_ABN_" + c) for c in abn_wide.columns if c != "HADM_ID"}
abn_wide = abn_wide.rename(abn_ren)

txt_wide = (
    text_latest
    .pivot(index="HADM_ID", on="TEST_KEY", values="VAL_TEXT", aggregate_function="first")
)
txt_ren = {c: ("LAB_TXT_" + c) for c in txt_wide.columns if c != "HADM_ID"}
txt_wide = txt_wide.rename(txt_ren)

# --- instead of full-join chaining, build a base key frame and left-join into it ---
base_keys = pl.concat(
    [
        num_wide.select("HADM_ID"),
        abn_wide.select("HADM_ID"),
        txt_wide.select("HADM_ID"),
    ],
    how="vertical_relaxed",
).unique()

wide = (
    base_keys
    .join(num_wide, on="HADM_ID", how="left")
    .join(abn_wide, on="HADM_ID", how="left")
    .join(txt_wide, on="HADM_ID", how="left")
)

print(
    f"[BUILD] Wide lab feature frame: {wide.height:,} rows, {wide.width} cols "
    f"(numeric:{max(num_wide.width-1,0)}, abn:{max(abn_wide.width-1,0)}, text:{max(txt_wide.width-1,0)})"
)

# merge with spine (left-join so no admissions are lost)
spine1 = spine.join(wide, on="HADM_ID", how="left")

# fill only the abnormal flags
fill_cols = [c for c in spine1.columns if c.startswith("LABABN__")]
if fill_cols:
    spine1 = spine1.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in fill_cols])


# Fill defaults: abnormal flags to 0; leave numeric/text as-is (missing = not measured)
fill_cols = [c for c in spine1.columns if c.startswith("LABABN__")]
if fill_cols:
    spine1 = spine1.with_columns([pl.col(c).fill_null(0).cast(pl.Int8) for c in fill_cols])

# ---------- 6) Output ----------
spine1.write_parquet(OUT_PQ, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PQ}")

# Small one-row report
summary = pl.DataFrame([{
    "n_rows": spine1.height,
    "n_numeric_labs": num_wide.width - 1,
    "n_text_labs": txt_wide.width - 1,
    "n_abn_flags": len(fill_cols),
    "pct_any_lab_abnormal": float(
        spine1.select(pl.max_horizontal(*fill_cols).mean() if fill_cols else pl.lit(0)).item()
    ) if fill_cols else 0.0,
}])
summary.write_csv(OUT_CSV)
print(f"[REPORT] Summary → {OUT_CSV}")
print("[DONE]")
