import json
import os
import time
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
from langgraph.types import Command
from langchain_core.load import dumps
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
    try:
        progress = get_job_progress(job_id)
        if progress:
            payload = {
                    "step": progress[2],
                    "status": progress[1]
                }
            yield f"data: {json.dumps(payload)}\n\n"
        last_heartbeat = time.time()
        while True:
            message = pubsub.get_message(ignore_subscribe_messages=True)
            if message:
                data = json.loads(message["data"])
                logger.debug(f"SSE send | {data}")
                yield f"data: {json.dumps(data)}\n\n"
                if data.get("status") == "completed":
                    yield f"data: {json.dumps(data)}\n\n"
                    time.sleep(0.5)
                    break

            # prevents nginx/browser closing idle connection
            if time.time() - last_heartbeat > 10:
                yield ":\n\n"   # SSE comment (no-op)
                last_heartbeat = time.time()

            time.sleep(0.1)

    except Exception as e:
        logger.error(f"SSE error | job_id={job_id} | error={str(e)}")
    finally:
        pubsub.close()
        logger.debug(f"Event stream closed | job_id={job_id}")




@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    logger.debug(f"Fetching result | job_id={job_id}")
    base_path = os.getenv("BOM_STORAGE_PATH", "/data/bom")
    file_path = os.path.join(base_path, f"{job_id}.json")
    if not os.path.exists(file_path):
         logger.warning(f"BOM file not found | job_id={job_id} | path={file_path}")
         raise HTTPException(status_code=404, detail="Result not ready")
  
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        bom = data.get("bom", [])
        logger.debug(f"BOM fetched | job_id={job_id} | count={len(bom)}")

        bom = enrich_bom_with_pricing(bom, MATERIAL_LOOKUP)
        logger.debug(f"Result ready | job_id={job_id} | count={len(bom)}")
        return {
            "job_id": job_id,
            "bom": bom
        }
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format | job_id={job_id}")
        raise HTTPException(status_code=500, detail="Corrupted result file")
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
            dumps({
                "step":None,
                "status": "processing"
            })
        )
        try:
            for thread_id, event in stream_estimation(job_id, file_path, output_dir):
                logger.info(f"STREAM EVENT 👉 {event}")

                if not isinstance(event, dict):
                    continue

                if "__interrupt__" in event:
                    logger.info("🚨 INTERRUPT DETECTED")
                    logger.info(f"RAW INTERRUPT 👉 {event['__interrupt__']}")
                    interrupt_obj = event["__interrupt__"]

                    if isinstance(interrupt_obj, tuple):
                        interrupt_obj = interrupt_obj[0]

                    review_data = interrupt_obj
                    while hasattr(review_data, "value"):
                        review_data = review_data.value
                    logger.info(f"✅ CLEAN REVIEW DATA 👉 {review_data}")
                    logger.info("📤 SENDING HITL TO FRONTEND")

                    redis_conn.redis_client.set(
                        f"hitl:{job_id}",
                        dumps(review_data)
                    )

                    redis_conn.redis_client.publish(
                        job_id,
                        dumps({
                            "step": "hitl_review",
                            "status": "waiting_for_user",
                            "data": review_data
                        })
                    )

                    return

            
                for node_name, _ in event.items():
                    redis_conn.redis_client.publish(
                        job_id,
                        dumps({
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

@router.get("/jobs/{job_id}/hitl")
def get_hitl(job_id: str):
    data = redis_conn.redis_client.get(f"hitl:{job_id}")
    
    if not data:
        raise HTTPException(404, "No HITL pending")

    return json.loads(data)


@router.post("/jobs/{job_id}/hitl")
def submit_hitl(job_id: str, payload: dict):

    resume_payload = {
        "corrected_bboxes": payload["corrected_bboxes"]
    }
    logger.info(f"🔄 RESUME STARTED | job_id={job_id}")
    logger.info(f"📥 RESUME PAYLOAD 👉 {resume_payload}")
    def resume():

        redis_conn.redis_client.delete(f"hitl:{job_id}")

        for thread_id, event in stream_estimation(
            job_id,
            None,
            None,
            command=Command(resume=resume_payload)
        ):
            logger.info(f"🔁 RESUME EVENT 👉 {event}")

            if not isinstance(event, dict):
                continue

            if "__interrupt__" in event:
                logger.info("🚨 INTERRUPT AGAIN AFTER RESUME")
                interrupt_obj = event["__interrupt__"]

                if isinstance(interrupt_obj, tuple):
                    interrupt_obj = interrupt_obj[0]

                review_data = interrupt_obj
                while hasattr(review_data, "value"):
                    review_data = review_data.value

                redis_conn.redis_client.set(
                    f"hitl:{job_id}",
                    dumps(review_data)
                )

                redis_conn.redis_client.publish(
                    job_id,
                    dumps({
                        "step": "hitl_review",
                        "status": "waiting_for_user",
                        "data": review_data
                    })
                )
                return

            for node_name, _ in event.items():
                logger.info(f"➡️ NEXT NODE AFTER RESUME 👉 {node_name}")
                redis_conn.redis_client.publish(
                    job_id,
                    dumps({
                        "step": node_name,
                        "status": "processing"
                    })
                )

    threading.Thread(target=resume).start()

    return {"message": "resumed"}

@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str):
    return StreamingResponse(
        event_generator(job_id),
        media_type="text/event-stream"
    )