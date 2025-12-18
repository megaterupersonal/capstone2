# 05_09b_final_preprocessing_linear.py
# Build linear-model-friendly dataset (Logistic, SVM, KNN)
# - Encodes LAB_TXT to categories
# - One-hot encodes selected categorical fields
# - Keeps continuous labs
# - Median-imputes numeric features
# - Standardizes continuous cols only
# - Removes leakage labels + unused IDs
# - Outputs final_linear_X.parquet + final_linear_y.csv

import os, pathlib
import polars as pl
from sklearn.preprocessing import StandardScaler
import numpy as np

FEAT_DIR  = r"H:\A\parquet files\features"
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

IN_PQ   = os.path.join(FEAT_DIR, "05_08_attach_labels.parquet")
OUT_XPQ = os.path.join(FEAT_DIR, "final_linear_features.parquet")
OUT_y   = os.path.join(FEAT_DIR, "final_linear_labels.csv")
OUT_FL  = os.path.join(FEAT_DIR, "final_linear_feature_list.txt")

LABEL_COL = "LBL_READMIT_30D"
ID_COLS   = ["HADM_ID"]

# -------------------- Load --------------------
df = pl.read_parquet(IN_PQ)
print(f"[LOAD] {df.height:,} rows, {df.width} columns")

# -------------------- Drop unnecessary columns --------------------
DROP_EXPLICIT = [
    "SUBJECT_ID",
    "LBL_READMIT_ELIGIBLE",
    "ICU_FIRST_INTIME", "ICU_LAST_OUTTIME",
    "TR_FIRST_INTIME", "TR_LAST_OUTTIME", "TR_FIRST_ICU_INTIME",
]

df = df.drop([c for c in DROP_EXPLICIT if c in df.columns])

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

# -------------------- Encode selected LAB_TXT columns -------------------- #Needs to be referenced in feature dictionary later
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

for col, mapping in TEXT_MAPPINGS.items():
    if col in df.columns:
        df = df.with_columns(
            pl.when(
                pl.col(col).is_null()
                | (pl.col(col).cast(pl.Utf8).str.strip_chars() == "")
            )
            .then(mapping.get("UNKNOWN", 0))
            .otherwise(
                pl.col(col)
                .cast(pl.Utf8)
                .str.to_uppercase()
                .replace(mapping, default=mapping.get("UNKNOWN", 0))
            )
            .alias(col.replace("LAB_TXT_", "LABTXT_"))
        )

# Drop original LAB_TXT_* columns
df = df.drop([c for c in df.columns if c.startswith("LAB_TXT_")])

# -------------------- One-hot encode categorical variables --------------------
# one-hot:
# - ADM_MONTH
# - ADM_WEEKDAY
# - Encoded LABTXT_ columns

categorical_cols = []
for c, t in df.schema.items():
    if c.startswith("LABTXT_"):
        categorical_cols.append(c)
    if c in ["ADM_MONTH", "ADM_WEEKDAY",
             "ADM_INSURANCE", "ADM_DISCHARGE_LOCATION",
             "ADM_ETHNICITY_GROUPED", "PAT_GENDER"]:
        categorical_cols.append(c)

if categorical_cols:
    df = df.to_dummies(columns=categorical_cols)
    print(f"[DUMMIES] One-hot encoded {len(categorical_cols)} base categorical cols → now {df.width} columns")

# -------------------- Numerical + Binary split --------------------
schema = df.schema

binary_cols = []
for c, t in schema.items():
    if c in ID_COLS or c == LABEL_COL:
        continue  
    if "Int" in str(t):
        try:
            nunique = df.select(pl.col(c).n_unique()).item()
            if nunique <= 3:
                binary_cols.append(c)
        except Exception:
            continue


continuous_cols = []
for c, t in schema.items():
    if c == LABEL_COL or c in ID_COLS:
        continue  
    if any(k in str(t) for k in ["Int", "UInt", "Float"]) and c not in binary_cols:
        continuous_cols.append(c)


print(f"[TYPES] Binary-like: {len(binary_cols)} | Continuous: {len(continuous_cols)}")

# -------------------- Impute --------------------
# Binary → fill 0
if binary_cols:
    df = df.with_columns([pl.col(c).fill_null(0) for c in binary_cols])

# Continuous → median
for c in continuous_cols:
    try:
        med = df.select(pl.col(c).median()).item()
        if med is not None:
            df = df.with_columns(pl.col(c).fill_null(med))
    except Exception:
        # skip weird columns if any
        continue

print("[IMPUTE] Completed binary (0) and continuous (median) imputation.")

# -------------------- Standardize continuous columns --------------------
if continuous_cols:
    scaler = StandardScaler()
    cont_np = df.select(continuous_cols).to_numpy()
    cont_scaled = scaler.fit_transform(cont_np)
    df = df.with_columns(
        [pl.Series(name=c, values=cont_scaled[:, i]) for i, c in enumerate(continuous_cols)]
    )
    print("[SCALE] Standardized continuous columns.")

# -------------------- Split X / y --------------------
id_present = any(col in df.columns for col in ID_COLS)
id_cols = [c for c in ID_COLS if c in df.columns]

if LABEL_COL not in df.columns:
    raise ValueError(f"Label column '{LABEL_COL}' is missing after processing.")

y = df.select(id_cols + [LABEL_COL])
X = df.drop(id_cols + [LABEL_COL])

# -------------------- Save --------------------
X.write_parquet(OUT_XPQ, compression="zstd")
y.write_csv(OUT_y)

with open(OUT_FL, "w", encoding="utf-8") as f:
    for c in X.columns:
        f.write(c + "\n")

print(f"[SAVE] X → {OUT_XPQ} ({X.height:,} rows × {X.width} cols)")
print(f"[SAVE] y → {OUT_y}")
print("[DONE]")
