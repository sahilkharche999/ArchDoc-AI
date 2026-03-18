import sqlite3
from datetime import datetime
from src.logger import setup_logger

logger = setup_logger(__name__)

DB_PATH = "checkpoints.sqlite"


def create_job(job_id: str, file_name: str):
    logger.info(f"Creating job | job_id={job_id} | file_name={file_name}")
    try:

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
        logger.info(f"Job created successfully | job_id={job_id}")
    except Exception as e:
        logger.error(
            f"Failed to create job | job_id={job_id} | error={str(e)}"
        )
        raise 
        
