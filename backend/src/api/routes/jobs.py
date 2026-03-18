import json

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from src.logger import setup_logger
from src.service import stream_estimation, app
from src.workflow.common.utils import enrich_bom_with_pricing
from src.workflow.common.utils import load_material_weights

logger = setup_logger(__name__)
router = APIRouter()
EXCEL_PATH = "#1A Steel Estimator (2023).xlsx"
MATERIAL_LOOKUP = load_material_weights(EXCEL_PATH)


def event_generator(job_id, file_path):
    logger.info(f"Event stream started | job_id={job_id} | file={file_path}")
    try:
        for thread_id, event in stream_estimation(job_id, file_path, "output_temp"):
            for node_name, state_update in event.items():
                logger.debug(f"Node update | job_id={job_id} | node={node_name}")
                payload = {
                    "node": node_name
                }
                yield f"data: {json.dumps(payload)}\n\n"
        logger.info(f"Event stream completed | job_id={job_id}")
    except Exception as e:
         logger.error(f"Event stream failed | job_id={job_id} | error={str(e)}")
         raise



@router.get("/jobs/stream")
def stream_job(job_id: str, file_path: str):
    logger.info(f"Stream request received | job_id={job_id} | file={file_path}")
    if not job_id:
        logger.error("Stream failed | reason=missing_job_id")
        raise HTTPException(status_code=400, detail="job_id required")
    logger.info(f"Streaming workflow for job {job_id}")
    return StreamingResponse(
        event_generator(job_id, file_path),
        media_type="text/event-stream"
    )


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    logger.info(f"Fetching result | job_id={job_id}")
    config = {"configurable": {"thread_id": job_id}}
    try:
        snapshot = app.get_state(config)

        if not snapshot.values:
            logger.error(f"Job not found | job_id={job_id}")
            raise HTTPException(status_code=404, detail="Job not found")

        bom_wrapper = snapshot.values.get("final_bill_of_materials", {})
        bom = bom_wrapper.get("final_bill_of_materials", [])
        logger.debug(f"BOM fetched | job_id={job_id} | count={len(bom)}")
        bom = enrich_bom_with_pricing(bom, MATERIAL_LOOKUP)
        logger.info(f"Result ready | job_id={job_id} | count={len(bom)}")
        return {
            "job_id": job_id,
            "bom": bom
        }
    except Exception as e:
        raise 
