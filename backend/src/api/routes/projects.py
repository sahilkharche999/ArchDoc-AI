from fastapi import APIRouter
from fastapi import HTTPException

from src.db.get_projects import get_projects as fetch_projects 
from src.db.get_projects import get_projects_by_id,get_job_progress,update_project,delete_project
from src.logger import setup_logger

router = APIRouter(prefix="/projects", tags=["projects"])
logger = setup_logger(__name__)


@router.get("/")
def get_projects():
    logger.debug("Fetch projects request received")
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
        logger.debug(f"Projects response ready | count={len(projects)}")
        return {"projects": projects}
    except Exception as e:
        logger.error(f"Failed to fetch projects | error={str(e)}")
        return { "projects": [] }


@router.get("/{job_id}")
def get_project(job_id: str):
    logger.debug(f"Fetching project from DB with ID: {job_id}")
    try:
        row = get_projects_by_id(job_id=job_id)
        progress_row=get_job_progress(job_id=job_id)
        if progress_row is None:
            current_step = "unknown"
            status = "processing"
        else:
            current_step = progress_row[2]
            status = progress_row[1]

        if row is None:
            logger.warning(f"Project not found | job_id={job_id}")
            raise HTTPException(status_code=404, detail="Project not found")
        logger.debug("Project fetched successfully")
        return {
            "job_id": row[0],
            "name": row[1],
            "status": status,
            "date": row[3],
            "file_path": f"assets/{row[0]}_structural.pdf",
            "current_step":current_step
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(e)}")
        raise


@router.put("/{job_id}")
def update_project_name(job_id: str, payload: dict):
    logger.debug(f"update project name for job ID: {job_id}")
    new_name = payload.get("new_name")
    try:
        update_project(job_id=job_id,new_name=new_name)
        logger.debug("Project fetched successfully")
        return {"message": "updated"}
    except HTTPException as h:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(h)}")
        raise
    except Exception as e:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(e)}")
        raise

@router.delete("/{job_id}")
def delete_project_by_id(job_id: str):
    logger.debug(f"delete project name for job ID: {job_id}")
    try:
        delete_project(job_id=job_id)
        logger.debug("Project deleted successfully")
        return {"message": "deleted"}
    except HTTPException as e:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(h)}")
        raise
    except Exception as e:
        logger.exception(f"Error in deleting project | job_id={job_id} | error={str(e)}")
        raise




