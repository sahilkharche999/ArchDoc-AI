from fastapi import APIRouter
import sqlite3

router = APIRouter()

@router.get("/projects")
def get_projects():

    conn = sqlite3.connect("checkpoints.sqlite")
    cursor = conn.cursor()

    rows = cursor.execute("""
        SELECT job_id, name, status, upload_date
        FROM jobs
        ORDER BY upload_date DESC
    """).fetchall()

    conn.close()

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