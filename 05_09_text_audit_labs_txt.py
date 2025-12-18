# 05_09_text_audit_labs_txt.py
# Audits selected LAB_TXT columns > ONE CSV with clearer uniqueness stats


import polars as pl
import os, pathlib, textwrap

# ---------- Paths ----------
FEAT_DIR  = r"H:\A\parquet files\features"
IN_PQ     = os.path.join(FEAT_DIR, "05_08_attach_labels.parquet")  # after labels step
OUT_CSV   = os.path.join(FEAT_DIR, "LABTXT_text_audit_summary.csv")
pathlib.Path(FEAT_DIR).mkdir(parents=True, exist_ok=True)

# ---------- Target columns ----------
LABTXT_COLS = [
    "LAB_TXT_BLOOD_INTUBATED",
    "LAB_TXT_BLOOD_VENTILATOR",
    "LAB_TXT_URINE_URINE_COLOR",
    "LAB_TXT_URINE_URINE_APPEARANCE",
    "LAB_TXT_URINE_NITRITE",
    "LAB_TXT_URINE_LEUKOCYTES",
    "LAB_TXT_URINE_BACTERIA",
    "LAB_TXT_URINE_YEAST",
    "LAB_TXT_URINE_BLOOD",
    "LAB_TXT_URINE_PROTEIN_MG_DL",
    "LAB_TXT_URINE_KETONE_MG_DL",
    "LAB_TXT_URINE_GLUCOSE_MG_DL",
    "LAB_TXT_URINE_RBC_HPF",
    "LAB_TXT_URINE_WBC_HPF",
    "LAB_TXT_URINE_EPITHELIAL_CELLS_HPF",
]
# ---------- Load ----------
df = pl.read_parquet(IN_PQ)
present_cols = [c for c in LABTXT_COLS if c in df.columns]
if not present_cols:
    raise ValueError("None of the LAB_TXT columns were found in the input frame.")

# ---------- Helpers ----------
def normalize_expr(e: pl.Expr) -> pl.Expr:
    # Uppercase, trim, collapse internal whitespace
    return (
        e.cast(pl.Utf8)
         .str.to_uppercase()
         .str.replace_all(r"\s+", " ", literal=False)
         .str.strip_chars()
    )

def summarize_text_col(df: pl.DataFrame, col: str) -> dict:
    s = df.get_column(col)

    # counts
    n_rows = df.height
    n_null = df.select(pl.col(col).null_count()).item()

    # normalize
    trimmed = df.select(normalize_expr(pl.col(col)).alias("_t")).get_column("_t")
    # identify blank tokens
    is_blank_mask = trimmed == ""
    has_blank = bool(trimmed.filter(is_blank_mask).len() > 0)

    # uniques (total vs non-empty)
    n_unique_total = int(trimmed.n_unique())
    clean_vals = trimmed.filter(~is_blank_mask)
    n_unique_nonempty = int(clean_vals.n_unique())

    # frequency for non-blanks
    if clean_vals.len() > 0:
        vc = (pl.DataFrame({"v": clean_vals})
                .group_by("v").len()
                .sort("len", descending=True))
        # top-5 values only (no counts in output unless you want them)
        top_vals = vc.select("v").head(5)["v"].to_list()
    else:
        top_vals = []

    top5_examples = " | ".join(top_vals) if top_vals else ""
    other_rare_count = max(n_unique_nonempty - len(top_vals), 0)

    return {
        "column": col,
        "n_rows": n_rows,
        "n_null": int(n_null),
        "n_unique": n_unique_total,             # includes blank if present
        "n_unique_nonempty": n_unique_nonempty, # excludes blank token
        "has_blank": int(has_blank),            # 1 if "" exists after trim
        "top5_examples": top5_examples,
        "other_rare_count": int(other_rare_count),
    }

# ---------- Build one CSV ----------
rows = [summarize_text_col(df, c) for c in present_cols]
out = pl.DataFrame(rows).sort("column")
out.write_csv(OUT_CSV)
print(f"[DONE] Wrote text audit → {OUT_CSV}")
