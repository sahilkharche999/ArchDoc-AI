from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_projects():
    logger.info("Fetching projects from DB")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT job_id, name, status, upload_date
            FROM jobs
            ORDER BY upload_date DESC
        """)
        rows = cursor.fetchall()
        cursor.close()
        logger.info(f"Projects fetched successfully | count={len(rows)}")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def get_projects_by_id(job_id: str):
    logger.info(f"Fetching project from DB with ID: {job_id}")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
        row = cursor.fetchone()
        cursor.close()
        logger.info(f"Project fetched successfully | job_id={job_id}")
        return row
    except Exception as e:
        logger.error(f"Failed to fetch project | job_id={job_id} | error={str(e)}")
        raise
    finally:
        release_conn(conn)