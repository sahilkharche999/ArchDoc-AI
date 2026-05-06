"""
Cleanup utility — wipes all artifacts for a job (or all jobs).
Usage:
    python cleanup.py --job-id <uuid>     # wipe one job
    python cleanup.py --all               # wipe everything
"""
import argparse
import os
import shutil
from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.db.connection import get_conn, release_conn
from src.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


def wipe_memgraph(job_id: str = None):
    """Delete graph nodes for one project, or everything."""
    driver = GraphDatabase.driver(os.getenv("NEO4J_URI"), auth=None)
    with driver.session() as s:
        if job_id:
            result = s.run("MATCH (n {project: $pid}) DETACH DELETE n", pid=job_id)
            logger.info(f"Memgraph: wiped nodes for project {job_id}")
    driver.close()


def wipe_files(job_id: str = None):
    """Delete output_temp, assets, and bom_storage for the job."""
    base_dirs = [
        os.getenv("OUTPUT_TEMP_DIR", "output_temp"),
        os.getenv("ASSETS_DIR", "assets"),
        os.getenv("BOM_STORAGE_PATH", "bom_storage"),
    ]
    for base in base_dirs:
        if not os.path.exists(base):
            continue
        if job_id:
            target = os.path.join(base, job_id)
            if os.path.exists(target):
                shutil.rmtree(target)
                logger.info(f"Filesystem: removed {target}")


def wipe_checkpoints(job_id: str = None):
    """Delete LangGraph checkpoint rows from postgres."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        if job_id:
            # LangGraph PostgresSaver uses these tables
            cur.execute("DELETE FROM checkpoints WHERE thread_id = %s", (job_id,))
            cur.execute("DELETE FROM checkpoint_writes WHERE thread_id = %s", (job_id,))
            cur.execute("DELETE FROM checkpoint_blobs WHERE thread_id = %s", (job_id,))
            logger.info(f"Checkpoints: wiped for {job_id}")
        conn.commit()
        cur.close()
    except Exception as e:
        logger.warning(f"Checkpoint wipe failed (tables may not exist yet): {e}")
        conn.rollback()
    finally:
        release_conn(conn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", help="Specific job to wipe")
    parser.add_argument("--all", action="store_true", help="Wipe everything")
    args = parser.parse_args()

    if not args.job_id and not args.all:
        parser.error("Specify --job-id <uuid> or --all")

    target = args.job_id if args.job_id else None
    confirm = input(
        f"⚠️  Wipe {'ALL data' if args.all else f'job {target}'}? [yes/no]: "
    )
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    wipe_memgraph(target)
    wipe_checkpoints(target)
    wipe_files(target)
    logger.info("✅ Cleanup complete")


if __name__ == "__main__":
    main()