# 04_1a_admissionsaudit.py
# This file audits demographic columns in the cleaned ADMISSIONS table.

import polars as pl
import os

# === Paths ===
PARQUET_DIR = r"H:\A\parquet files\sampled"
FILE_PATH = os.path.join(PARQUET_DIR, "ADMISSIONS_cleaned.parquet")

# === Load parquet ===
ad = pl.read_parquet(FILE_PATH)
print(f"[LOAD] ADMISSIONS_cleaned: {ad.shape[0]:,} rows, {ad.shape[1]} columns")

# === Columns to audit ===
demo_cols = ["RELIGION", "MARITAL_STATUS", "LANGUAGE", "ETHNICITY", "INSURANCE"]

# === Define “unknown-like” strings ===
unknown_like = ["UNKNOWN", "NOT SPECIFIED", "UNOBTAINABLE" "UNKNOWN/NOT SPECIFIED", "NA", "N/A", "OTHER", "DECLINED TO ANSWER"]

records = []

for col in demo_cols:
    total = ad.height
    # nulls
    n_nulls = ad.filter(pl.col(col).is_null()).height
    # unknowns (case-insensitive)
    n_unknown_like = (
        ad.filter(
            pl.col(col)
            .cast(pl.Utf8)
            .str.to_uppercase()
            .is_in([u.upper() for u in unknown_like])
        )
        .height
    )
    # % unknown-like
    pct_unknown_like = round(n_unknown_like / total * 100, 2)

    n_unique = ad.select(pl.col(col).n_unique()).item()

    # top 5 values (readable)
    top5_df = (
        ad.group_by(col)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(5)
    )
    top5_str = "; ".join(
        f"{str(v)}: {int(c)}"
        for v, c in zip(top5_df[col].to_list(), top5_df["count"].to_list())
    )

    records.append({
        "column": col,
        "n_rows": total,
        "n_nulls": n_nulls,
        "n_unknownlike": n_unknown_like,
        "pct_unknownlike": pct_unknown_like,
        "n_unique": n_unique,
        "top_5_values": top5_str,
    })

summary = pl.DataFrame(records)

# === Output ===
out_csv = os.path.join(PARQUET_DIR, "ADMISSIONS_demographics_audit.csv")
summary.write_csv(out_csv)
print(summary)
print(f"\n[REPORT] Wrote → {out_csv}")
