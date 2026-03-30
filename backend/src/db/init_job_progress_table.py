from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)

def init_job_progress_table():
    logger.info("Initializing jobs progress table")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs_progress (
                job_id TEXT PRIMARY KEY,
                status TEXT,
                current_state TEXT
            )
        """)
        conn.commit()
        cursor.close()
        logger.info("Jobs progress table ready (created or already exists)")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize jobs table | error={str(e)}")
        raise
    finally:
        release_conn(conn)