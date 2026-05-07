from fastapi import APIRouter
from fastapi import HTTPException
import threading
from src.db.get_projects import get_projects as fetch_projects 
from src.db.get_projects import get_projects_by_id,get_job_progress,update_project,delete_project
from src.logger import setup_logger
from src.cleanup import wipe_memgraph,wipe_files,wipe_checkpoints
from src.api.routes.jobs import cancelled_jobs
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
            "date": row[4],
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

def cleanup_all(job_id: str):
    try:
        wipe_memgraph(job_id)
        wipe_checkpoints(job_id)
        wipe_files(job_id)
    except Exception as e:
        print(f"Cleanup failed for {job_id}: {str(e)}")

@router.delete("/{job_id}")
def delete_project_by_id(job_id: str):
    logger.debug(f"delete project name for job ID: {job_id}")
    try:
        cancelled_jobs.add(job_id) 
        delete_project(job_id=job_id)
        threading.Thread(target=cleanup_all,args=(job_id,)).start()

        logger.debug("Project deleted successfully")
        return {"message": "deleted and cleanup started"}
    except HTTPException as e:
        logger.error(f"Error fetching project | job_id={job_id} | error={str(h)}")
        raise
    except Exception as e:
        logger.exception(f"Error in deleting project | job_id={job_id} | error={str(e)}")
        raise




