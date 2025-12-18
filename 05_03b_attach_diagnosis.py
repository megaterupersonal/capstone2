# 05_03b_attach_diagnosis.py — Elixhauser with DX_ prefixes
# This file attaches Elixhauser comorbidity flags to the main spine table.
#In medicine, Dx is a common abbreviation for diagnosis.

import os, pathlib
import polars as pl

# ---------- Paths ----------
CLEAN_DIR = r"H:\A\parquet files\cleaned"
FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

SPINE_IN  = os.path.join(FEAT_DIR, "05_02_attach_transfers.parquet")
ELIX_PATH = os.path.join(FEAT_DIR, "elix_flags.parquet")

OUT_PARQUET = os.path.join(FEAT_DIR, "05_03_attach_diagnosis.parquet")
OUT_SUMMARY = os.path.join(FEAT_DIR, "05_03_attach_diagnosis_summary.csv")

# ---------- Load ----------
if not os.path.exists(SPINE_IN):
    raise FileNotFoundError(f"Spine input file not found: {SPINE_IN}")
if not os.path.exists(ELIX_PATH):
    raise FileNotFoundError(f"Elixhauser flags file not found: {ELIX_PATH}")

spine = pl.read_parquet(SPINE_IN).with_columns(pl.col("HADM_ID").cast(pl.Int64))
elix  = pl.read_parquet(ELIX_PATH).with_columns(pl.col("HADM_ID").cast(pl.Int64))

print(f"[LOAD] Spine: {os.path.basename(SPINE_IN)}  → {spine.height:,} rows, {spine.width} cols")
print(f"[LOAD] Elix flags: {os.path.basename(ELIX_PATH)} → {elix.height:,} rows, {elix.width} cols")

# ---------- Identify Elixhauser flags ----------
skip_cols = {"HADM_ID", "ELIX_VW_SCORE", "ELIX_N_FLAGS", "ELIX_ANY"}
raw_flag_cols = [c for c in elix.columns if c not in skip_cols]

# Rename all flag columns → DX_*
rename_map = {c: f"DX_{c.upper()}" for c in raw_flag_cols}
elix = elix.rename(rename_map)
dx_flag_cols = [rename_map[c] for c in raw_flag_cols]
elix = elix.with_columns([pl.col(c).cast(pl.Int8) for c in dx_flag_cols])

# ---------- Compute Van Walraven score if missing ----------
if "ELIX_VW_SCORE" not in elix.columns:
    vw_weights_raw = {
        "chf": 7, "carit": 5, "valv": 0, "pcd": 4, "pvd": 2,
        "hypunc": -1, "hypc": 0, "para": 7, "ond": 6, "cpd": 3,
        "diabunc": 0, "diabc": 0, "hypothy": -2, "rf": 5, "ld": 11,
        "pud": 0, "aids": 0, "lymph": 9, "metacanc": 12, "solidtum": 4,
        "rheumd": 0, "coag": 3, "obes": -5, "wloss": 6, "fed": 5,
        "blane": -2, "dane": -2, "alcohol": 0, "drug": 0, "psycho": 0, "depre": -3
    }
    weights = {f"DX_{k.upper()}": w for k, w in vw_weights_raw.items() if f"DX_{k.upper()}" in elix.columns}
    if not weights:
        raise ValueError("No recognized DX_* columns found to compute VW score.")
    elix = elix.with_columns(
        pl.sum_horizontal([pl.col(c) * float(w) for c, w in weights.items()])
          .cast(pl.Float64)
          .alias("DX_ELIX_VW_SCORE")
    )
else:
    elix = elix.rename({"ELIX_VW_SCORE": "DX_ELIX_VW_SCORE"})

# ---------- Add burden features ----------
# (1) number of diagnosis flags
elix = elix.with_columns(
    pl.sum_horizontal([pl.col(c) for c in dx_flag_cols])
      .cast(pl.Int16)
      .alias("DX_ELIX_N_FLAGS")
)
# (2) any comorbidity present
elix = elix.with_columns(
    (pl.col("DX_ELIX_N_FLAGS") > 0).cast(pl.Int8).alias("DX_ELIX_ANY")
)

# ---------- Join into spine ----------
n0 = spine.select(pl.col("HADM_ID").n_unique()).item()
spine1 = spine.join(elix, on="HADM_ID", how="left")

# Fill nulls for missing diagnoses
fill_cols = [*dx_flag_cols, "DX_ELIX_N_FLAGS", "DX_ELIX_ANY"]
for c in fill_cols:
    if c in spine1.columns:
        spine1 = spine1.with_columns(pl.col(c).fill_null(0))
if "DX_ELIX_VW_SCORE" in spine1.columns:
    spine1 = spine1.with_columns(pl.col("DX_ELIX_VW_SCORE").fill_null(0.0))

n1 = spine1.select(pl.col("HADM_ID").n_unique()).item()
assert n0 == n1, f"[ERROR] HADM_ID count changed after join: before={n0}, after={n1}"

# ---------- Summary ----------
def _safe_item(df: pl.DataFrame, expr: pl.Expr):
    out = df.select(expr)
    return None if out.height == 0 or out.width == 0 else out.item()

summary_df = pl.DataFrame([{
    "n_rows": spine1.height,
    "pct_any_dx": _safe_item(spine1, pl.col("DX_ELIX_ANY").mean().round(4)),
    "mean_n_dx_flags": _safe_item(spine1, pl.col("DX_ELIX_N_FLAGS").mean().round(3)),
    "mean_vw_score": _safe_item(spine1, pl.col("DX_ELIX_VW_SCORE").mean().round(3)),
    "median_vw_score": _safe_item(spine1, pl.col("DX_ELIX_VW_SCORE").median().round(3)),
    "p90_vw_score": _safe_item(spine1, pl.col("DX_ELIX_VW_SCORE").quantile(0.9, "nearest").round(3)),
}])

summary_df.write_csv(OUT_SUMMARY)
print(f"[REPORT] Wrote → {OUT_SUMMARY}")

spine1.write_parquet(OUT_PARQUET, compression="zstd")
print(f"[SAVE] Wrote → {OUT_PARQUET}")
print("[DONE]")
