import sqlite3


def update_job_status(job_id: str, status: str):
    DB_PATH = "checkpoints.sqlite"
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE jobs
        SET status = ?
        WHERE job_id = ?
        """,
        (status, job_id)
    )

    conn.commit()
    conn.close()
