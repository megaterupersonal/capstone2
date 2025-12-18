#08_00_extract_single_patient.py
# Extract single patient data row from final feature + label datasets (a tested row)

import os
import polars as pl
import json

# -----------------------------
# USER INPUT
# -----------------------------
TARGET_HADM_ID = 145121   # <<< CHANGE THIS FOR ANOTHER PATIENT
FEATURE_TYPE   = "tree"   # "tree" or "linear"
# -----------------------------

FEAT_DIR  = r"H:\A\parquet files\features"
LLM_DIR  = r"H:\A\parquet files\features\llm demo"

if FEATURE_TYPE == "tree":
    X_PATH = os.path.join(FEAT_DIR, "final_tree_features.parquet")
    y_PATH = os.path.join(FEAT_DIR, "final_tree_labels.csv")
elif FEATURE_TYPE == "linear":
    X_PATH = os.path.join(FEAT_DIR, "final_linear_features.parquet")
    y_PATH = os.path.join(FEAT_DIR, "final_linear_labels.csv")
else:
    raise ValueError("FEATURE_TYPE must be 'tree' or 'linear'.")

LABEL_COL = "LBL_READMIT_30D"
ID_COL    = "HADM_ID"

print(f"[LOAD] Features: {X_PATH}")
X = pl.read_parquet(X_PATH)

print(f"[LOAD] Labels:   {y_PATH}")
y = pl.read_csv(y_PATH)

if ID_COL not in y.columns:
    raise ValueError(f"ERROR: '{ID_COL}' not found in {y_PATH}.")

# ------------------------------------------------------------------
# Attach original row index BEFORE filtering so we know index in X
# ------------------------------------------------------------------
y_with_idx = y.with_row_index("ROW_IDX")

matches = y_with_idx.filter(pl.col(ID_COL) == TARGET_HADM_ID)

if matches.height == 0:
    print(f"[RESULT] No rows found for HADM_ID = {TARGET_HADM_ID}")
    raise SystemExit

if matches.height > 1:
    print(f"[WARN] Found {matches.height} rows for HADM_ID {TARGET_HADM_ID}; "
          f"using the first one.")

# Get the original row index (position in X / y)
idx = int(matches["ROW_IDX"][0])
label_row = matches.drop("ROW_IDX").row(0, named=True)

print(f"[FOUND] HADM_ID {TARGET_HADM_ID} at original row index {idx}")

# extract matching feature row from X
feat_row = X.row(idx, named=True)

# merge feature + label dictionaries
full_row = {**feat_row, **label_row}

# ---------- Display ----------
print("\n==============================")
print(f" Patient {TARGET_HADM_ID} — Features + Label")
print("==============================")
for k, v in full_row.items():
    print(f"{k:40s}: {v}")

# ---------- Save to JSON ----------
os.makedirs(LLM_DIR, exist_ok=True)
OUT_PATH = os.path.join(LLM_DIR, f"patient_{TARGET_HADM_ID}.json")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(full_row, f, indent=2)

print(f"\n[SAVED] JSON saved → {OUT_PATH}")
