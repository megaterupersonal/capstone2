# 03_1_othertables_to_parquet.py
# This file converts additional CSV tables to Parquet format using DuckDB.

import os, pathlib, duckdb

RAW_DIR = r"H:\A\csv files"
PQ_DIR  = r"H:\A\parquet files"

RAW_DIR = os.path.expanduser(RAW_DIR)
PQ_DIR  = os.path.expanduser(PQ_DIR)
pathlib.Path(PQ_DIR).mkdir(parents=True, exist_ok=True)

con = duckdb.connect(database=':memory:')
con.execute(f"PRAGMA threads = {os.cpu_count()};")
con.execute("PRAGMA enable_progress_bar = true;")

def csv_exists(name):
    return os.path.exists(os.path.join(RAW_DIR, f"{name}.csv"))

def copy_select_to_parquet(select_sql: str, out_name: str):
    print(f"[LOAD] {out_name}.parquet")
    out_path = os.path.join(PQ_DIR, f"{out_name}.parquet")
    con.execute(f"COPY ({select_sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD);")

if csv_exists("DIAGNOSES_ICD"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, icd9_code, seq_num
        FROM read_csv_auto('{os.path.join(RAW_DIR, "DIAGNOSES_ICD.csv")}', SAMPLE_SIZE=50000)
        """,
        "DIAGNOSES_ICD"
    )
    print("[OK] DIAGNOSES.parquet created successfully.")

if csv_exists("D_ICD_DIAGNOSES"):
    copy_select_to_parquet(
        f"SELECT * FROM read_csv_auto('{os.path.join(RAW_DIR, "D_ICD_DIAGNOSES.csv")}')",
        "D_ICD_DIAGNOSES"
    )
    print("[OK] D_ICD_DIAGNOSES.parquet created successfully.")

if csv_exists("LABEVENTS"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, itemid, charttime, value, valuenum, valueuom, flag
        FROM read_csv_auto('{os.path.join(RAW_DIR, "LABEVENTS.csv")}', SAMPLE_SIZE=50000,
                           DATEFORMAT='%Y-%m-%d %H:%M:%S')
        """,
        "LABEVENTS"
    )
    print("[OK] LABEVENTS.parquet created successfully.")

if csv_exists("D_LABITEMS"):
    copy_select_to_parquet(
        f"SELECT * FROM read_csv_auto('{os.path.join(RAW_DIR, "D_LABITEMS.csv")}')",
        "D_LABITEMS"
    )
    print("[OK] D_LABITEMS.parquet created successfully.")

if csv_exists("ICUSTAYS"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, icustay_id, intime, outtime, los
        FROM read_csv_auto('{os.path.join(RAW_DIR, "ICUSTAYS.csv")}',
                           DATEFORMAT='%Y-%m-%d %H:%M:%S')
        """,
        "ICUSTAYS"
    )
    print("[OK] ICUSTAYS.parquet created successfully.")

if csv_exists("TRANSFERS"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, icustay_id, dbsource, eventtype,
               prev_careunit, curr_careunit, intime, outtime
        FROM read_csv_auto('{os.path.join(RAW_DIR, "TRANSFERS.csv")}',
                           DATEFORMAT='%Y-%m-%d %H:%M:%S')
        """,
        "TRANSFERS"
    )
    print("[OK] TRANSFERS.parquet created successfully.")

if csv_exists("PROCEDURES_ICD"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, icd9_code, seq_num
        FROM read_csv_auto('{os.path.join(RAW_DIR, "PROCEDURES_ICD.csv")}')
        """,
        "PROCEDURES_ICD"
    )
    print("[OK] PROCEDURES.parquet created successfully.")

if csv_exists("D_ICD_PROCEDURES"):
    copy_select_to_parquet(
        f"SELECT * FROM read_csv_auto('{os.path.join(RAW_DIR, 'D_ICD_PROCEDURES.csv')}')",
        "D_ICD_PROCEDURES"
    )
    print("[OK] D_ICD_PROCEDURES.parquet created successfully.")

if csv_exists("PRESCRIPTIONS"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, drug, startdate, enddate, dose_val_rx, dose_unit_rx
        FROM read_csv_auto('{os.path.join(RAW_DIR, "PRESCRIPTIONS.csv")}',
                           DATEFORMAT='%Y-%m-%d %H:%M:%S')
        """,
        "PRESCRIPTIONS"
    )
    print("[OK] PRESCRIPTIONS.parquet created successfully.")

if csv_exists("MICROBIOLOGYEVENTS"):
    copy_select_to_parquet(
        f"""
        SELECT subject_id, hadm_id, chartdate, charttime, org_name, ab_name, interpretation
        FROM read_csv_auto('{os.path.join(RAW_DIR, "MICROBIOLOGYEVENTS.csv")}',
                           DATEFORMAT='%Y-%m-%d %H:%M:%S')
        """,
        "MICROBIOLOGYEVENTS"
    )
    print("[OK] MICROBIOLOGY.parquet created successfully.")

print("Done converting selected tables to Parquet.")
