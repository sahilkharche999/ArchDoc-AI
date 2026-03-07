import sqlite3

conn = sqlite3.connect("checkpoints.sqlite")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    name TEXT,
    file_name TEXT,
    status TEXT,
    upload_date TEXT
)
""")

conn.commit()
conn.close()

print("Jobs table created successfully")