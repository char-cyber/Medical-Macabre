
import sqlite3
import pandas as pd

# create database
conn = sqlite3.connect("mydata.db").+

# load CSVs
df1 = pd.read_csv("NOTEEVENTS.csv")
df2 = pd.read_csv("DIAGNOSES_ICD.csv")

# write to SQLite
df1.to_sql("notes", conn, if_exists="replace", index=False)
df2.to_sql("diagnoses", conn, if_exists="replace", index=False)

# run join
query = """
SELECT n.ROW_ID, n.TEXT
FROM notes n
INNER JOIN diagnoses d
ON n.HADM_ID = d.HADM_ID
WHERE n.CATEGORY = 'Discharge summary';

"""

merged = pd.read_sql_query(query, conn)

# save result
merged.to_csv("merged.csv", index=False)

conn.close()