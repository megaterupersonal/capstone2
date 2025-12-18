#02_1_sample_creation.py
# This file creates a sampled subset of patients for modeling, balancing positives and negatives.

import os, duckdb, pathlib

PARQUET_DIR = r"H:\A\parquet files"
OUT_DIR = r"H:\A\parquet files\samples"

PARQUET_DIR = os.path.expanduser(PARQUET_DIR)
OUT_DIR = os.path.expanduser(OUT_DIR)
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

con = duckdb.connect(database=':memory:')
con.execute(f"PRAGMA threads = {os.cpu_count()};")
con.execute("PRAGMA enable_progress_bar = true;")

# register Parquet tables
con.execute(f"CREATE VIEW PATIENTS   AS SELECT * FROM read_parquet('{os.path.join(PARQUET_DIR, 'PATIENTS.parquet')}');")
con.execute(f"CREATE VIEW ADMISSIONS AS SELECT * FROM read_parquet('{os.path.join(PARQUET_DIR, 'ADMISSIONS.parquet')}');")

# 1) Label 30-day readmissions 
con.execute("""
CREATE OR REPLACE TABLE adm_labels AS
WITH ordered AS (
  SELECT
    subject_id, hadm_id, admittime, dischtime, admission_type,
    LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime)      AS next_admittime,
    LEAD(admission_type) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_admtype
  FROM ADMISSIONS
)
SELECT
  subject_id, hadm_id, admittime, dischtime, admission_type,
  CASE WHEN next_admittime IS NOT NULL
            AND date_diff('day', dischtime, next_admittime) <= 30
            AND coalesce(next_admtype, '') <> 'ELECTIVE'
       THEN 1 ELSE 0 END AS readmit30
FROM ordered;
""")

#Persist labels to Parquet for later auditing
con.execute(f"COPY adm_labels TO '{os.path.join(OUT_DIR, 'adm_labels.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD);")

# 2) Identify positive patients (any readmit30 == 1)
con.execute("CREATE OR REPLACE TEMP TABLE pos_patients AS SELECT DISTINCT subject_id FROM adm_labels WHERE readmit30 = 1;")

# 3) Build patient universe and find negative-only patients
con.execute("CREATE OR REPLACE TEMP TABLE all_patients AS SELECT DISTINCT subject_id FROM ADMISSIONS;")
con.execute("""
CREATE OR REPLACE TEMP TABLE neg_only AS
SELECT a.subject_id
FROM all_patients a
LEFT JOIN pos_patients p USING(subject_id)
WHERE p.subject_id IS NULL;
""")

# 4) Sample 20% of negative-only patients 
con.execute("""
CREATE OR REPLACE TEMP TABLE neg_sample AS
SELECT *
FROM neg_only
WHERE (hash(subject_id) % 100) < 20;
""")

# 5) Sampled patients = ALL positives + sampled negatives
con.execute("""
CREATE OR REPLACE TABLE sampled_patients AS
SELECT subject_id FROM pos_patients
UNION ALL
SELECT subject_id FROM neg_sample;
""")

# 6) Sampled keys (subject_id, hadm_id) for all admissions of selected patients
con.execute("""
CREATE OR REPLACE TABLE sampled_keys AS
SELECT DISTINCT a.subject_id, a.hadm_id
FROM ADMISSIONS a
JOIN sampled_patients c USING(subject_id);
""")

# 7) Persist outputs
con.execute(f"COPY sampled_patients TO '{os.path.join(OUT_DIR, 'sampled_patients.parquet')}' (FORMAT PARQUET, COMPRESSION ZSTD);")
con.execute(f"COPY sampled_keys     TO '{os.path.join(OUT_DIR, 'sampled_keys.parquet')}'     (FORMAT PARQUET, COMPRESSION ZSTD);")

# quick CSVs for inspection
con.execute(f"COPY (SELECT * FROM sampled_patients LIMIT 20) TO '{os.path.join(OUT_DIR, 'sampled_patients_head.csv')}' WITH (HEADER, DELIMITER ',');")
con.execute(f"COPY (SELECT * FROM sampled_keys LIMIT 20)     TO '{os.path.join(OUT_DIR, 'sampled_keys_head.csv')}'     WITH (HEADER, DELIMITER ',');")

print("Samples built. Files written to:", OUT_DIR)
