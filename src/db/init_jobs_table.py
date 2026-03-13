import sqlite3

DB_PATH = "checkpoints.sqlite"

def init_jobs_table():
    conn = sqlite3.connect(DB_PATH)
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