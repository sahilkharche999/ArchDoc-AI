from datetime import datetime

from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)


def create_job(job_id: str, file_name: str):
    logger.info(f"Creating job | job_id={job_id} | file_name={file_name}")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO jobs (job_id, name, file_name, status, upload_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            job_id,
            file_name,
            file_name,
            "Processing",
            datetime.now().strftime("%Y-%m-%d")
        ))
        conn.commit()
        cursor.close()
        logger.info(f"Job created successfully | job_id={job_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to create job | job_id={job_id} | error={str(e)}")
        raise
    finally:
        release_conn(conn)