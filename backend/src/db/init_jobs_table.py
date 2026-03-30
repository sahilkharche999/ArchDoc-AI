from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)


def init_jobs_table():
    logger.debug("Initializing jobs table")
    conn = get_conn()
    try:
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
        cursor.close()
        logger.debug("Jobs table ready (created or already exists)")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize jobs table | error={str(e)}")
        raise
    finally:
        release_conn(conn)