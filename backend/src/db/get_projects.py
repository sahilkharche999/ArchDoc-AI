from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

logger = setup_logger(__name__)


def get_projects():
    logger.debug("Fetching projects from DB")
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
        logger.debug(f"Projects fetched successfully | count={len(rows)}")
        return rows
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def get_projects_by_id(job_id: str):
    logger.debug(f"Fetching project from DB with ID: {job_id}")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
        row = cursor.fetchone()
        cursor.close()
        logger.debug(f"Project fetched successfully | job_id={job_id}")
        return row
    except Exception as e:
        logger.error(f"Failed to fetch project | job_id={job_id} | error={str(e)}")
        raise
    finally:
        release_conn(conn)


def get_job_progress(job_id: str):
    logger.debug(f"Fetching project from jobs_progress DB with ID: {job_id}")
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM jobs_progress WHERE job_id = %s", (job_id,))
        row = cursor.fetchone()
        cursor.close()
        logger.debug(f"Project fetched successfully | job_id={job_id}")
        return row
    except Exception as e:
        logger.error(f"Failed to fetch project | job_id={job_id} | error={str(e)}")
        raise
    finally:
        release_conn(conn)





def update_project(job_id: str, new_name: str):
    logger.debug(f"[DB] Update project | job_id={job_id} | new_name={new_name}")
    conn = get_conn()
    try:
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE jobs
            SET name = %s
            WHERE job_id = %s
        """, (new_name,job_id))

        conn.commit()
        cursor.close()
        logger.debug(f"[DB] Project updated | job_id={job_id}")
    except Exception as e:
        conn.rollback()
        logger.error(
            f"[DB] Update project failed | job_id={job_id} | error={str(e)}"
        )
        raise

    finally:
        release_conn(conn)


def delete_project(job_id: str):
    conn = get_conn()
    logger.debug(f"[DB] Delete project | job_id={job_id}")
    try:
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM jobs WHERE job_id = %s",
            (job_id,)
        )
        cursor.execute(
            "DELETE FROM jobs_progress WHERE job_id = %s",
            (job_id,)
        )

        conn.commit()
        cursor.close()
        logger.debug(f"[DB] Project deleted | job_id={job_id}")

    except Exception as e:
        conn.rollback()
        logger.error(
            f"[DB] Delete project failed | job_id={job_id} | error={str(e)}"
        )
        raise

    finally:
        release_conn(conn)