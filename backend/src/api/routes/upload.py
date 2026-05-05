import os
import uuid

from fastapi import APIRouter, UploadFile, File, Form
from fastapi import HTTPException
from pypdf import PdfReader, PdfWriter
from typing import Optional
from src.db.create_job import create_job,create_job_progress
from src.logger import setup_logger
import src.redis_conn as redis_conn

logger = setup_logger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


@router.post("")
async def upload_file(
        file: UploadFile = File(...),
        start_page: int = Form(...),
        end_page: int = Form(...),
        sheet_prefix: Optional[str] = Form(default="")
):
    upload_dir = os.getenv("ASSETS_DIR", "/data/assets")
    os.makedirs(upload_dir, exist_ok=True)
    logger.debug(
        f"Upload request received | filename={file.filename} | start={start_page} | end={end_page}"
    )

    job_id = str(uuid.uuid4())

    file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")

    with open(file_path, "wb") as f:
        f.write(await file.read())

    reader = PdfReader(file_path)
    writer = PdfWriter()
    total_pages = len(reader.pages)

    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise HTTPException(status_code=400, detail="Invalid page range")

    for i in range(start_page - 1, end_page):
        writer.add_page(reader.pages[i])

    trimmed_path = os.path.join(upload_dir, f"{job_id}_structural.pdf")

    with open(trimmed_path, "wb") as f:
        writer.write(f)

    display_name = os.path.splitext(file.filename)[0]
    create_job(job_id, display_name)
    create_job_progress(job_id)
    if sheet_prefix:
        redis_conn.redis_client.set(f"sheet_prefix:{job_id}", sheet_prefix.strip().upper())

    os.remove(file_path)
    return {
        "job_id": job_id,
        "file_path": trimmed_path,
        "sheet_prefix": sheet_prefix
    }
