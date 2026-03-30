from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)


def update_job_status(job_id: str, status: str):
    logger.info(f"Updating job status | job_id={job_id} | status={status}")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE jobs
            SET status = %s
            WHERE job_id = %s
        """, (status, job_id))
        conn.commit()
        cursor.close()
        logger.info(f"Job status updated successfully | job_id={job_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update job status | job_id={job_id} | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def update_job_progress(job_id:str, status:str, step:str):
    logger.info(f"Updating job | job_id={job_id} | status={status} ")
    conn = get_conn()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE jobs_progress
            SET current_state = %s, status = %s
            WHERE job_id = %s
        """, (step,status,job_id))

        conn.commit()
        cursor.close()

        logger.info(f"Job updated successfully | job_id={job_id}")

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to update job | job_id={job_id} | error={str(e)}")
        raise

    finally:
        release_conn(conn)
