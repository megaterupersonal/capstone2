## 02_2_testintegrity.py
# This file runs basic tests to ensure data integrity after processing steps.

import duckdb, os
out = r"H:\A\parquet files"
con = duckdb.connect()

# counts
print(con.execute(f"SELECT COUNT(*) AS n_adm FROM '{out}/adm_labels.parquet'").fetchdf())
print(con.execute(f"SELECT COUNT(DISTINCT subject_id) AS n_sampled_patients FROM '{out}/sampled_patients.parquet'").fetchdf())
print(con.execute(f"SELECT COUNT(*) AS n_sampled_keys, COUNT(DISTINCT hadm_id) AS n_hadm FROM '{out}/sampled_keys.parquet'").fetchdf())

# positives vs negatives (overall)
print(con.execute(f"""
SELECT readmit30, COUNT(*) AS n
FROM '{out}/adm_labels.parquet'
GROUP BY 1
ORDER BY 1 DESC
""").fetchdf())

# ensure sampled_keys only includes sampled patients
print(con.execute(f"""
SELECT COUNT(*) AS not_in_sampled
FROM '{out}/sampled_keys.parquet' k
LEFT JOIN '{out}/sampled_patients.parquet' p USING(subject_id)
WHERE p.subject_id IS NULL
""").fetchdf())

# sanity: elective next admissions are not counted as positives
print(con.execute(f"""
WITH ordered AS (
  SELECT subject_id, hadm_id, admittime,
         LEAD(admission_type) OVER (PARTITION BY subject_id ORDER BY admittime) AS next_type
  FROM '{out}/adm_labels.parquet'
)
SELECT COUNT(*) AS positives_with_next_elective
FROM '{out}/adm_labels.parquet' a
JOIN ordered o USING(subject_id, hadm_id)
WHERE a.readmit30=1 AND o.next_type='ELECTIVE'
""").fetchdf())
