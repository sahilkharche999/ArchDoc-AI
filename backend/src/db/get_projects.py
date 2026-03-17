import sqlite3

DB_PATH = "checkpoints.sqlite"


def get_projects():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT job_id, name, status, upload_date
        FROM jobs
        ORDER BY upload_date DESC
    """).fetchall()

    conn.close()

    return rows
