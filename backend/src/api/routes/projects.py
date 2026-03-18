from fastapi import APIRouter
from fastapi import HTTPException

from src.db.get_projects import get_projects as fetch_projects
from src.logger import setup_logger

router = APIRouter(prefix="/projects", tags=["projects"])
logger = setup_logger(__name__)


@router.get("/")
def get_projects():
    logger.info("Fetch projects request received")
    try:

        rows = fetch_projects()
        logger.debug(f"Fetched projects from DB | count={len(rows)}")

        projects = [
            {
                "job_id": r[0],
                "name": r[1],
                "status": r[2],
                "date": r[3]
            }
            for r in rows
        ]
        logger.info(f"Projects response ready | count={len(projects)}")

        return {"projects": projects}
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        raise


@router.get("/{job_id}")
def get_project(job_id: str):
    logger.info(f"Fetch single project request | job_id={job_id}")
    try:
        rows = fetch_projects()
        logger.debug(f"Fetched projects for lookup | count={len(rows)}")

        for r in rows:
            if r[0] == job_id:
                logger.info(f"Project found | job_id={job_id}")
                return {
                    "job_id": r[0],
                    "name": r[1],
                    "status": r[2],
                    "date": r[3],
                    "file_path": f"assets/{r[0]}_structural.pdf"
                }
        logger.error(f"Project not found | job_id={job_id}")
        raise HTTPException(status_code=404, detail="Project not found")
    except Exception as e:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(e)}")
        raise
