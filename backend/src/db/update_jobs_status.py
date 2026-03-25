import sqlite3
from src.logger import setup_logger

logger = setup_logger(__name__)

def update_job_status(job_id: str, status: str):
    logger.info(f"Updating job status | job_id={job_id} | status={status}")
    try:

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
        logger.info(f"Job status updated successfully | job_id={job_id}")
    except Exception as e:
        logger.error(
            f"Failed to update job status | job_id={job_id} | error={str(e)}"
        )
        raise

