import sqlite3
from datetime import datetime

DB_PATH = "checkpoints.sqlite"

def create_job(job_id: str, file_name: str):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO jobs (job_id, name, file_name, status, upload_date)
        VALUES (?, ?, ?, ?, ?)
    """, (
        job_id,
        file_name,
        file_name,
        "Processing",
        datetime.now().strftime("%Y-%m-%d")
    ))

    conn.commit()
    conn.close()