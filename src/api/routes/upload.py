from fastapi import APIRouter, UploadFile, File
import sqlite3
import os
import uuid
from src.workflow.common.logger import setup_logger
from datetime import datetime
logger = setup_logger(__name__)

router = APIRouter()

UPLOAD_DIR = "assets"

def create_job(job_id, file_name):

    conn = sqlite3.connect("checkpoints.sqlite")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO jobs (job_id, name, file_name, status, upload_date)
        VALUES (?, ?, ?, ?, ?)
    """, (
        job_id,
        file_name,
        file_name,
        "Processing",
        datetime.now().strftime("%Y-%m-%d")
    ))

    conn.commit()
    conn.close()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info("api call to upload the ")

    job_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    create_job(job_id, file.filename)

    return {
        "job_id": job_id,
        "file_path": file_path
    }

