from fastapi import APIRouter
from fastapi import HTTPException

from src.db.get_projects import get_projects as fetch_projects 
from src.db.get_projects import get_projects_by_id
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
        logger.exception(f"Failed to fetch projects | error={str(e)}")
        return { "projects": [] }


@router.get("/{job_id}")
def get_project(job_id: str):
    logger.info(f"Fetching project from DB with ID: {job_id}")
    try:
        row = get_projects_by_id(job_id=job_id)
        if row is None:
            logger.warning(f"Project not found | job_id={job_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        logger.info("Project fetched successfully")
        return {
            "job_id": row[0],
            "name": row[1],
            "status": row[2],
            "date": row[3],
            "file_path": f"assets/{row[0]}_structural.pdf"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching project | job_id={job_id} | error={str(e)}")
        raise
