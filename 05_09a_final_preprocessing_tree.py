# 05_09a_final_preprocessing_tree.py
# Build tree-friendly training table (adds LAB_TXT mapping based on audit)

import os, pathlib
import polars as pl

FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

IN_PQ   = os.path.join(FEAT_DIR, "05_08_attach_labels.parquet")
OUT_XPQ = os.path.join(FEAT_DIR, "final_tree_features.parquet")
OUT_y   = os.path.join(FEAT_DIR, "final_tree_labels.csv")
OUT_FL  = os.path.join(FEAT_DIR, "final_tree_feature_list.txt")
OUT_SM  = os.path.join(FEAT_DIR, "final_tree_summary.csv")

LABEL_COL = "LBL_READMIT_30D"

# ---------- Load ----------
df = pl.read_parquet(IN_PQ)
print(f"[LOAD] {os.path.basename(IN_PQ)} → {df.height:,} rows, {df.width} cols")

# ---------- Sanity: label presence ----------
if LABEL_COL not in df.columns:
    candidates = [c for c in df.columns if "READMIT" in c.upper()]
    raise ValueError(f"Missing label column: {LABEL_COL}. Found these: {candidates}")

# ---------- Columns to drop (explicit leakage and IDs) ----------
drop_explicit = [
    "SUBJECT_ID",
    "LBL_READMIT_ELIGIBLE",
    "ICU_FIRST_INTIME", "ICU_LAST_OUTTIME", "ADM_DISCH_TO_FACILITY",
    "TR_FIRST_INTIME", "TR_LAST_OUTTIME", "TR_FIRST_ICU_INTIME", "LBL_DAYS_TO_NEXT_ADMIT"
]

# Drop raw columns before one-hot
drop_explicit = [c for c in drop_explicit if c in df.columns]
if drop_explicit:
    df = df.drop(drop_explicit)
    print(f"[CLEAN] Dropped columns: {drop_explicit}")

# ---------- Drop discharge-related leakage (including one-hot columns) ----------
leakage_prefixes = [
    "ADM_DISCHARGE_LOCATION",   # catches main + one-hot dummy cols
]

leakage_cols = [c for c in df.columns
                if any(c.startswith(prefix) for prefix in leakage_prefixes)]

if leakage_cols:
    df = df.drop(leakage_cols)
    print(f"[CLEAN] Dropped discharge leakage columns: {leakage_cols}")

# ---------- Apply selected LAB_TXT mappings ----------
# These mappings come from your audit summary (labtxt_text_audit_summary.txt)
TEXT_MAPPINGS = {
    "LAB_TXT_BLOOD_INTUBATED": {"INTUBATED": 1, "NOT INTUBATED": 0},
    "LAB_TXT_BLOOD_VENTILATOR": {"CONTROLLED": 2, "SPONTANEOUS": 1, "IMV": 3, "UNKNOWN": 0},
    "LAB_TXT_URINE_BACTERIA": {"NONE": 0, "RARE": 1, "FEW": 2, "OCC": 3, "MOD": 4, "ABUNDANT": 5, "UNKNOWN": 0},
    "LAB_TXT_URINE_BLOOD": {"NEG": 0, "TR": 1, "SM": 2, "MOD": 3, "LG": 4, "UNKNOWN": 0},
    "LAB_TXT_URINE_EPITHELIAL_CELLS_HPF": {"0-2": 0, "3-5": 1, "6-10": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_GLUCOSE_MG_DL": {"NEG": 0, "POS": 1, "UNKNOWN": 0},
    "LAB_TXT_URINE_KETONE_MG_DL": {"NEG": 0, "TR": 1, "UNKNOWN": 0},
    "LAB_TXT_URINE_LEUKOCYTES": {"NEG": 0, "TR": 1, "SM": 2, "MOD": 3, "POS": 4, "UNKNOWN": 0},
    "LAB_TXT_URINE_NITRITE": {"NEG": 0, "POS": 1, "UNKNOWN": 0},
    "LAB_TXT_URINE_PROTEIN_MG_DL": {"NEG": 0, "TR": 1, "POS": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_RBC_HPF": {"0-2": 0, "3-5": 1, "6-10": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_URINE_APPEARANCE": {"CLEAR": 0, "HAZY": 1, "CLOUDY": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_URINE_COLOR": {"YELLOW": 0, "STRAW": 1, "AMBER": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_WBC_HPF": {"0-2": 0, "3-5": 1, "6-10": 2, "UNKNOWN": 0},
    "LAB_TXT_URINE_YEAST": {"NONE": 0, "RARE": 1, "FEW": 2, "UNKNOWN": 0},
}

mapped_cols = []
for col, mapping in TEXT_MAPPINGS.items():
    if col in df.columns:
        expr = (
            pl.when(pl.col(col).is_null() | (pl.col(col).str.strip_chars() == ""))
              .then(pl.lit(mapping.get("UNKNOWN", 0)))
              .otherwise(
                  pl.col(col)
                    .cast(pl.Utf8)
                    .str.to_uppercase()
                    .str.strip_chars()
                    .replace(mapping, default=mapping.get("UNKNOWN", 0))
              )
              .alias(col.replace("LAB_TXT_", "LABTXT_"))
        )
        df = df.with_columns(expr)
        mapped_cols.append(col.replace("LAB_TXT_", "LABTXT_"))

if mapped_cols:
    print(f"[ENCODE] Encoded {len(mapped_cols)} LAB_TXT columns → {mapped_cols[:5]}{'...' if len(mapped_cols) > 5 else ''}")

# ---------- Auto-drop by type/prefix ----------
date_cols = [c for c, t in df.schema.items() if ("Datetime" in str(t) or "Date" in str(t))]
txt_cols = [c for c in df.columns if c.startswith("LAB_TXT_")]  # old text cols only
auto_drop = list(set(date_cols + txt_cols))
if auto_drop:
    df = df.drop(auto_drop)
    print(f"[CLEAN] Auto-dropped {len(auto_drop)} non-numeric/text cols")

# Hard-drop any label-like or future columns
drop_future = [c for c in df.columns if c.upper().startswith("LBL_") and c != "LBL_READMIT_30D"]
drop_future += ["LBL_DAYS_TO_NEXT_ADMIT"]  
drop_future = list(dict.fromkeys([c for c in drop_future if c in df.columns]))

if drop_future:
    df = df.drop(drop_future)
    print(f"[CLEAN] Dropped future/label-like cols: {drop_future}")

# ---------- Keep only numeric dtypes ----------
numeric_kinds = ("Int", "UInt", "Float")
keep_numeric = [c for c, t in df.schema.items() if any(k in str(t) for k in numeric_kinds)]
if LABEL_COL not in keep_numeric:
    df = df.with_columns(pl.col(LABEL_COL).cast(pl.Int8))
    keep_numeric = [c for c, t in df.schema.items() if any(k in str(t) for k in numeric_kinds)]
df_num = df.select(keep_numeric)

# ---------- Imputation ----------
print("[IMPUTE] Starting numeric imputation...")

# Heuristic classification
schema = df_num.schema
binary_like = [
    c for c, t in schema.items()
    if ("Int8" in str(t)) or (df_num.select(pl.col(c).drop_nulls().n_unique()).item() <= 3)
]
continuous_like = [
    c for c, t in schema.items()
    if (any(k in str(t) for k in ["Float", "Int"])) and (c not in binary_like)
]

# Binary imputation: fill with 0
if binary_like:
    df_num = df_num.with_columns([pl.col(c).fill_null(0) for c in binary_like])

# Continuous imputation: fill with median (only among observed)
for c in continuous_like:
    try:
        median_val = df_num.select(pl.col(c).median()).item()
        if median_val is not None:
            df_num = df_num.with_columns(pl.col(c).fill_null(median_val))
    except Exception:
        # Some columns may be entirely null — skip safely
        continue

print(f"[IMPUTE] Filled {len(binary_like)} binary-like cols with 0 and {len(continuous_like)} continuous cols with median.")


# ---------- Split X / y ----------
hadm_present = "HADM_ID" in df_num.columns
id_col = ["HADM_ID"] if hadm_present else []

y = df_num.select(id_col + [LABEL_COL])
X = df_num.drop([LABEL_COL] + id_col)

# ---------- Save ----------
X.write_parquet(OUT_XPQ, compression="zstd")
y.write_csv(OUT_y)
with open(OUT_FL, "w", encoding="utf-8") as f:
    for c in X.columns:
        f.write(c + "\n")

n_rows, n_feats = X.height, X.width
pos_rate = y.select(pl.col(LABEL_COL).mean()).item() if n_rows else 0.0
summary = pl.DataFrame([{
    "n_rows": n_rows,
    "n_features": n_feats,
    "label_pos_rate": round(float(pos_rate), 4),
    "dropped_auto_n": len(auto_drop),
    "encoded_labtxt_n": len(mapped_cols),
}])
summary.write_csv(OUT_SM)

print(f"[SAVE] X → {OUT_XPQ} ({n_rows:,}×{n_feats})")
print(f"[SAVE] y → {OUT_y}")
print(f"[REPORT] Encoded LABTXT cols: {len(mapped_cols)}")
print("[DONE]")
