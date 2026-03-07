import sqlite3

def update_job_status(job_id: str, status: str):

    conn = sqlite3.connect("checkpoints.sqlite")
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