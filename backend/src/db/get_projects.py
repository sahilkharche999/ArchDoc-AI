import sqlite3
from src.logger import setup_logger

logger = setup_logger(__name__)
DB_PATH = "checkpoints.sqlite"


def get_projects():
    logger.info("Fetching projects from DB")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        rows = cursor.execute("""
            SELECT job_id, name, status, upload_date
            FROM jobs
            ORDER BY upload_date DESC
        """).fetchall()

        conn.close()
        logger.info(f"Projects fetched successfully | count={len(rows)}")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        raise 

def get_projects_by_id(job_id:str):
    logger.info("Fetching projects from DB with ID : {job_id}")
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        rows = cursor.execute("SELECT * FROM jobs WHERE job_id = ?",(job_id,)).fetchone()

        conn.close()
        logger.info(f"Projects fetched successfully | count={len(rows)}")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        raise 
    