#01_csv_to_parquet.py
# This file converts CSV files to Parquet format using DuckDB for efficient storage and retrieval.
#Github version: paths are omitted

import os, duckdb, pathlib

#paths 
RAW_DIR = r"H:\A\csv files"         # <- change me (folder with PATIENTS.csv, ADMISSIONS.csv)
PARQUET_DIR = r"H:\A\parquet files" # <- change me

RAW_DIR = os.path.expanduser(RAW_DIR)
PARQUET_DIR = os.path.expanduser(PARQUET_DIR)
pathlib.Path(PARQUET_DIR).mkdir(parents=True, exist_ok=True)

#duckdb session
con = duckdb.connect(database=':memory:')
# use all cores available
con.execute(f"PRAGMA threads = {os.cpu_count()};")
con.execute("PRAGMA enable_progress_bar = true;")

#helper to convert a CSV to Parquet quickly
def csv_to_parquet(csv_name, table_cols=None):
    csv_path = os.path.join(RAW_DIR, f"{csv_name}.csv")
    pq_path  = os.path.join(PARQUET_DIR, f"{csv_name}.parquet")

    con.execute(f"""
        CREATE OR REPLACE VIEW V_{csv_name} AS
        SELECT * FROM read_csv_auto('{csv_path}', DATEFORMAT='%Y-%m-%d %H:%M:%S', SAMPLE_SIZE=50000);
    """)
    con.execute(f"COPY (SELECT * FROM V_{csv_name}) TO '{pq_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")
    print(f"Converted {csv_name}.csv -> {csv_name}.parquet")

csv_to_parquet("PATIENTS")
csv_to_parquet("ADMISSIONS")

print("Done.")
