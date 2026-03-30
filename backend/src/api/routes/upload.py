import os
import uuid

from fastapi import APIRouter, UploadFile, File, Form
from fastapi import HTTPException
from pypdf import PdfReader, PdfWriter

from src.db.create_job import create_job,create_job_progress
from src.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = "assets"


@router.post("")
async def upload_file(
        file: UploadFile = File(...),
        start_page: int = Form(...),
        end_page: int = Form(...)
):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(
        f"Upload request received | filename={file.filename} | start={start_page} | end={end_page}"
    )

    job_id = str(uuid.uuid4())

    file_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    reader = PdfReader(file_path)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise HTTPException(status_code=400, detail="Invalid page range")

    for i in range(start_page - 1, end_page):
        writer.add_page(reader.pages[i])

    trimmed_path = os.path.join(UPLOAD_DIR, f"{job_id}_structural.pdf")

    with open(trimmed_path, "wb") as f:
        writer.write(f)

    display_name = os.path.splitext(file.filename)[0]
    create_job(job_id, display_name)
    create_job_progress(job_id)

    os.remove(file_path)
    return {
        "job_id": job_id,
        "file_path": trimmed_path
    }
