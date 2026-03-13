from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi import HTTPException
import json
from src.service import stream_estimation,app
from src.logger import setup_logger
from src.workflow.common.utils import load_material_weights
from src.workflow.common.utils import enrich_bom_with_pricing

logger = setup_logger(__name__)
router = APIRouter()
EXCEL_PATH = "#1A Steel Estimator (2023).xlsx"
MATERIAL_LOOKUP = load_material_weights(EXCEL_PATH)

def event_generator(job_id,file_path):
    for thread_id, event in stream_estimation(job_id,file_path, "output_temp"):
        for node_name, state_update in event.items():
            payload = {
                "node": node_name
            }
            yield f"data: {json.dumps(payload)}\n\n"

@router.get("/jobs/stream")
def stream_job(job_id: str,file_path: str):
    if not job_id:
     raise HTTPException(status_code=400, detail="job_id required")
    logger.info(f"Streaming workflow for job {job_id}")
    return StreamingResponse(
        event_generator(job_id,file_path),
        media_type="text/event-stream"
    )

@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    config = {"configurable": {"thread_id": job_id}}

    snapshot = app.get_state(config)
    
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Job not found")

    bom_wrapper = snapshot.values.get("final_bill_of_materials", {})
    bom = bom_wrapper.get("final_bill_of_materials", [])
    bom = enrich_bom_with_pricing(bom, MATERIAL_LOOKUP)
    return {
        "job_id": job_id,
        "bom": bom
    }
