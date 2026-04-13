from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import json
from src.logger import setup_logger
from src.workflow.common.utils import normalize_pdf_orientation

router = APIRouter(prefix="/pdf", tags=["pdf"])
logger = setup_logger(__name__)

@router.post("/fix")
async def fix_pdf(
    file: UploadFile = File(...),
    rotation_map: str = Form(...)
):
    try:
        temp_dir = os.getenv("TEMP_DIR", "/tmp")
        os.makedirs(temp_dir, exist_ok=True)

        job_id = str(uuid.uuid4())

        input_path = os.path.join(temp_dir, f"{job_id}_input.pdf")
        output_path = os.path.join(temp_dir, f"{job_id}_fixed.pdf")

        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # Parse rotation map
        try:
            page_angles = json.loads(rotation_map)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid rotation_map JSON")

        # Convert keys to int
        page_angles = {int(k): int(v) for k, v in page_angles.items()}
        logger.info(f"Rotation map received: {page_angles}")
        logger.info(f"Input file: {input_path}")

        # Apply rotation
        normalize_pdf_orientation(input_path, output_path, page_angles)

        return FileResponse(
            output_path,
            media_type="application/pdf",
            filename=file.filename
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))