from fastapi import APIRouter
from fastapi import HTTPException

from src.db.get_projects import get_projects as fetch_projects
from src.logger import setup_logger

router = APIRouter(prefix="/projects", tags=["projects"])
logger = setup_logger(__name__)


@router.get("/")
def get_projects():
    rows = fetch_projects()

    projects = [
        {
            "job_id": r[0],
            "name": r[1],
            "status": r[2],
            "date": r[3]
        }
        for r in rows
    ]

    return {"projects": projects}


@router.get("/{job_id}")
def get_project(job_id: str):
    rows = fetch_projects()

    for r in rows:
        if r[0] == job_id:
            return {
                "job_id": r[0],
                "name": r[1],
                "status": r[2],
                "date": r[3],
                "file_path": f"assets/{r[0]}_structural.pdf"
            }

    raise HTTPException(status_code=404, detail="Project not found")
