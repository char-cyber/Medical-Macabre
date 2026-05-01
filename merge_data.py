
import duckdb

con = duckdb.connect("mimic_join.duckdb")

con.execute("""
COPY (
    SELECT n.*, d.*
    FROM read_csv_auto('NOTEEVENTS.csv') AS n
    INNER JOIN read_csv_auto('DIAGNOSES_ICD.csv') AS d
    ON n.HADM_ID = d.HADM_ID
)
TO 'joined_output.csv'
WITH (HEADER, DELIMITER ',');
""")

print("Finished joining CSV files.")