## 03_2_tablepruning.py

import os, pathlib, duckdb

PARQUET_DIR = r"H:\A\parquet files"     # source Parquet folder
OUTPUT_DIR = r"H:\A\parquet files\samples"  # output folder for pruned tables

PARQUET_DIR = os.path.expanduser(PARQUET_DIR)
OUTPUT_DIR  = os.path.expanduser(OUTPUT_DIR)
pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

LAB_ITEMID_WHITELIST = [
    # Examples: Na (50983), K (50971), Creatinine (50912), WBC (51301/51300 variants),
    # Leave list empty for now, maybe once feature engineering is done and find out what to filter on
    # 50983, 50971, 50912, 51301, 51300, 50868, 50902, 50931, 50960, 50820 
    # (THIS MIGHT GO UNUSED)
]


# DUCKDB SESSION

con = duckdb.connect(database=':memory:')
con.execute(f"PRAGMA threads = {os.cpu_count()};")
con.execute("PRAGMA enable_progress_bar = true;")

def pq(path):  # shorthand
    return os.path.join(PARQUET_DIR, path)

def out(path):
    return os.path.join(OUTPUT_DIR, path)

def parquet_exists(name):
    return os.path.exists(pq(f"{name}.parquet"))

# Verify Parquet write by counting rows before and after
def copy_and_verify(out_name: str, sql: str, *, source_name: str = None):
    """
    Writes the result of `sql` to pruned/{out_name}.parquet, then verifies:
      expected = COUNT(*) of the SQL result (pre-write), so when its loaded into memory
      actual   = COUNT(*) read back from the written Parquet (post-write), so how many output rows are actually there
    Also prints the % retained vs full source table if `source_name` is given.
    """
    out_path = out(f"{out_name}.parquet")

    # Write Parquet
    con.execute(f"COPY ({sql}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"[OK] {out_name}.parquet")

    # Expected rows from the SELECT
    exp = con.execute(f"SELECT COUNT(*) AS n FROM ({sql}) t").fetchone()[0]

    # Actual rows in the written Parquet
    act = con.execute(f"SELECT COUNT(*) AS n FROM read_parquet('{out_path}')").fetchone()[0]

    # % retained vs full source
    pct_txt = ""
    if source_name:
        # count full source rows (unfiltered)
        src_pq = pq(f"{source_name}.parquet")
        if os.path.exists(src_pq):
            src_total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{src_pq}')").fetchone()[0]
            if src_total:
                pct = (act / src_total) * 100.0
                pct_txt = f" | retained ~{pct:.2f}% of {source_name}"

    status = "OK" if exp == act else "MISMATCH"
    print(f"[VERIFY] {out_name}: expected={exp:,} actual={act:,} [{status}]{pct_txt}")


# REGISTER sampled ARTIFACTS

if not os.path.exists(out("..")):  
    pass

if not os.path.exists(os.path.join(PARQUET_DIR, "ADMISSIONS.parquet")):
    raise FileNotFoundError("ADMISSIONS.parquet not found in PARQUET_DIR.")

sampled_patients = os.path.join(PARQUET_DIR, "sampled_patients.parquet")
sampled_keys     = os.path.join(PARQUET_DIR, "sampled_keys.parquet")

# fallbacks
if not os.path.exists(sampled_patients) or not os.path.exists(sampled_keys):
    raise FileNotFoundError("sampled_patients.parquet or sampled_keys.parquet not found in PARQUET_DIR.")

con.execute(f"CREATE VIEW sampled_PATIENTS AS SELECT * FROM read_parquet('{sampled_patients}')")
con.execute(f"CREATE VIEW sampled_KEYS     AS SELECT * FROM read_parquet('{sampled_keys}')")


# PRUNE ADMISSIONS & PATIENTS

try:
    if parquet_exists("ADMISSIONS"):
        sql = f"""
        SELECT a.*
        FROM read_parquet('{pq("ADMISSIONS.parquet")}') a
        JOIN sampled_KEYS c USING(subject_id, hadm_id)
        """
        copy_and_verify("ADMISSIONS_sampled", sql, source_name="ADMISSIONS")
    if parquet_exists("PATIENTS"):
        sql = f"""
        SELECT p.*
        FROM read_parquet('{pq("PATIENTS.parquet")}') p
        JOIN sampled_PATIENTS c USING(subject_id)
        """
        copy_and_verify("PATIENTS_sampled", sql, source_name="PATIENTS")
except Exception as e:
    print(f"[ERROR] Pruning ADMISSIONS/PATIENTS failed: {e}")


# PRUNE DIAGNOSES (with dictionary labels)
try:
    if parquet_exists("DIAGNOSES_ICD"):
        con.execute(f"CREATE VIEW DIAG AS SELECT * FROM read_parquet('{pq('DIAGNOSES_ICD.parquet')}')")
        # if dictionary available, join it for labels. it's done for the other stuff as well with dictionaries.
        if parquet_exists("D_ICD_DIAGNOSES"):
            con.execute(f"CREATE VIEW DDIAG AS SELECT * FROM read_parquet('{pq('D_ICD_DIAGNOSES.parquet')}')")
            sql = """
            SELECT d.subject_id, d.hadm_id, d.icd9_code, d.seq_num,
                   m.short_title, m.long_title
            FROM DIAG d
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            LEFT JOIN DDIAG m USING(icd9_code)
            """
            #joining with dictionary to get labels
        else:
            sql = """
            SELECT d.subject_id, d.hadm_id, d.icd9_code, d.seq_num
            FROM DIAG d
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            """
            # no dictionary available
        copy_and_verify("DIAGNOSES_ICD_sampled", sql, source_name="DIAGNOSES_ICD")
except Exception as e:
    print(f"[ERROR] Pruning DIAGNOSES failed: {e}")


# PRUNE PROCEDURES (with dictionary)
try:
    if parquet_exists("PROCEDURES_ICD"):
        con.execute(f"CREATE VIEW PROC AS SELECT * FROM read_parquet('{pq('PROCEDURES_ICD.parquet')}')")
        if parquet_exists("D_ICD_PROCEDURES"):
            con.execute(f"CREATE VIEW DPROC AS SELECT * FROM read_parquet('{pq('D_ICD_PROCEDURES.parquet')}')")
            sql = """
            SELECT p.subject_id, p.hadm_id, p.icd9_code, p.seq_num,
                   d.short_title, d.long_title
            FROM PROC p
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            LEFT JOIN DPROC d USING(icd9_code)
            """
        else:
            sql = """
            SELECT p.subject_id, p.hadm_id, p.icd9_code, p.seq_num
            FROM PROC p
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            """
        copy_and_verify("PROCEDURES_ICD_sampled", sql, source_name="PROCEDURES_ICD")
except Exception as e:
    print(f"[ERROR] Pruning PROCEDURES failed: {e}")

# PRUNE LABEVENTS (with dictionary)
try:
    if parquet_exists("LABEVENTS"):
        con.execute(f"CREATE VIEW LAB AS SELECT * FROM read_parquet('{pq('LABEVENTS.parquet')}')")
        lab_filter = ""
        if LAB_ITEMID_WHITELIST:
            # Make a CSV string of itemids for SQL IN (...)
            ids = ",".join(str(x) for x in LAB_ITEMID_WHITELIST)
            lab_filter = f"AND LAB.itemid IN ({ids})"

        if parquet_exists("D_LABITEMS"):
            con.execute(f"CREATE VIEW DLAB AS SELECT * FROM read_parquet('{pq('D_LABITEMS.parquet')}')")
            sql = f"""
            SELECT LAB.subject_id, LAB.hadm_id, LAB.itemid, LAB.charttime,
                   LAB.value, LAB.valuenum, LAB.valueuom, LAB.flag,
                   DLAB.label, DLAB.category, DLAB.fl uid
            FROM LAB
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            LEFT JOIN DLAB USING(itemid)
            WHERE 1=1 {lab_filter}
            """
        else:
            sql = f"""
            SELECT LAB.subject_id, LAB.hadm_id, LAB.itemid, LAB.charttime,
                   LAB.value, LAB.valuenum, LAB.valueuom, LAB.flag
            FROM LAB
            JOIN sampled_KEYS c USING(subject_id, hadm_id)
            WHERE 1=1 {lab_filter}
            """
        copy_and_verify("LABEVENTS_sampled", sql, source_name="LABEVENTS")
except Exception as e:
    print(f"[ERROR] Pruning LABEVENTS failed: {e}")


# PRUNE ICUSTAYS & TRANSFERS
try:
    if parquet_exists("ICUSTAYS"):
        sql = f"""
        SELECT i.subject_id, i.hadm_id, i.icustay_id, i.intime, i.outtime, i.los
        FROM read_parquet('{pq("ICUSTAYS.parquet")}') i
        JOIN sampled_KEYS c USING(subject_id, hadm_id)
        """
        copy_and_verify("ICUSTAYS_sampled", sql, source_name="ICUSTAYS")
    if parquet_exists("TRANSFERS"):
        sql = f"""
        SELECT t.subject_id, t.hadm_id, t.icustay_id, t.dbsource, t.eventtype,
               t.prev_careunit, t.curr_careunit, t.intime, t.outtime
        FROM read_parquet('{pq("TRANSFERS.parquet")}') t
        JOIN sampled_KEYS c USING(subject_id, hadm_id)
        """
        copy_and_verify("TRANSFERS_sampled", sql, source_name="TRANSFERS")
except Exception as e:
    print(f"[ERROR] Pruning ICUSTAYS/TRANSFERS failed: {e}")


# PRUNE PRESCRIPTIONS
try:
    if parquet_exists("PRESCRIPTIONS"):
        sql = f"""
        SELECT subject_id, hadm_id, drug, startdate, enddate, dose_val_rx, dose_unit_rx
        FROM read_parquet('{pq("PRESCRIPTIONS.parquet")}')
        JOIN sampled_KEYS USING(subject_id, hadm_id)
        """
        copy_and_verify("PRESCRIPTIONS_sampled", sql, source_name="PRESCRIPTIONS")  
except Exception as e:
    print(f"[ERROR] Pruning PRESCRIPTIONS failed: {e}")

# PRUNE MICROBIOLOGYEVENTS
try:
    if parquet_exists("MICROBIOLOGYEVENTS"):
        sql = f"""
        SELECT subject_id, hadm_id, chartdate, charttime, org_name, ab_name, interpretation
        FROM read_parquet('{pq("MICROBIOLOGYEVENTS.parquet")}')
        JOIN sampled_KEYS USING(subject_id, hadm_id)
        """
        copy_and_verify("MICROBIOLOGYEVENTS_sampled", sql, source_name="MICROBIOLOGYEVENTS")
except Exception as e:
    print(f"[ERROR] Pruning MICROBIOLOGYEVENTS failed: {e}")

print("Done. Pruned tables written to:", OUTPUT_DIR)
