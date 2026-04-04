import json
import os
import threading
from fastapi import APIRouter
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
import src.redis_conn as redis_conn
from src.logger import setup_logger
from src.service import stream_estimation, app
from src.workflow.common.utils import enrich_bom_with_pricing
from src.workflow.common.utils import load_material_weights
from src.db.update_jobs_status import update_job_progress
from src.db.get_projects import get_job_progress
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

class StartJobRequest(BaseModel):
    job_id: str
    file_path: str


logger = setup_logger(__name__)
router = APIRouter()
EXCEL_PATH = os.getenv("EXCEL_PATH", "Steel Estimator.xlsx")
MATERIAL_LOOKUP = load_material_weights(EXCEL_PATH)

def event_generator(job_id:str):

    logger.debug(f"Event stream started | job_id={job_id} ")
    pubsub = redis_conn.redis_client.pubsub()
    pubsub.subscribe(job_id)
    progress = get_job_progress(job_id)
    if progress:
        payload = {
            "step": progress[2],
            "status": progress[1]
        }
        yield f"data: {json.dumps(payload)}\n\n"


    try:
        for message in pubsub.listen():
            if message["type"] != "message":
                continue
            data = json.loads(message["data"])
            yield f"data: {json.dumps(data)}\n\n"
            if data.get("status") == "completed":
                break
    except Exception as e:
        logger.error(f"SSE error | job_id={job_id} | error={str(e)}")
    finally:
        pubsub.close()
        logger.debug(f"Event stream closed | job_id={job_id}")




@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    logger.debug(f"Fetching result | job_id={job_id}")
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
        logger.debug(f"Result ready | job_id={job_id} | count={len(bom)}")
        return {
            "job_id": job_id,
            "bom": bom
        }
    except Exception as e:
        logger.exception(f"Failed to fetch result | job_id={job_id}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/jobs/start")
def start_job(request: StartJobRequest):
    
    job_id = request.job_id
    file_path = request.file_path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(
    os.path.join(BASE_DIR, "../../../"))
    output_dir = os.path.join(PROJECT_ROOT, "output_temp", job_id)

    existing = get_job_progress(job_id)

    if existing and existing[1] == "processing":
        return {"message": "already running"}

    def run():
        logger.info(f"Job started | job_id={job_id}")
        update_job_progress(job_id, "processing",None)
        redis_conn.redis_client.publish(
            job_id,
            json.dumps({
                "step":None,
                "status": "processing"
            })
        )
        try:
            for thread_id, event in stream_estimation(job_id, file_path,output_dir ):
                for node_name, state_update in event.items():
                    
                    update_job_progress(job_id, "processing", node_name)
                    redis_conn.redis_client.publish(
                    job_id,
                    json.dumps({
                        "step": node_name,
                        "status": "processing"
                    })
                )

            update_job_progress(job_id, "completed", "agent_4_merger")
            redis_conn.redis_client.publish(
                job_id,
                json.dumps({
                    "step": "agent_4_merger",
                    "status": "completed"
                })
            )
            logger.info(f"Job completed successfully | job_id={job_id}")

        except Exception as e:
            logger.error(f"Processing failed | job_id={job_id} | error={str(e)}")

    threading.Thread(target=run).start()

    return {"message": "started"}


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str):
    return StreamingResponse(
        event_generator(job_id),
        media_type="text/event-stream"
    )