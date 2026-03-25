import sqlite3
from src.logger import setup_logger

logger = setup_logger(__name__)
DB_PATH = "checkpoints.sqlite"


def init_jobs_table():
    logger.info(f"Initializing jobs table | db_path={DB_PATH}")
    try:

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
        logger.info("Jobs table ready (created or already exists)")
    except Exception as e:
        logger.error(f"Failed to initialize jobs table | error={str(e)}")
        raise 
