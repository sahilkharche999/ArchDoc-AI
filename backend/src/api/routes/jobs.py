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
cancelled_jobs = set()

def event_generator(job_id:str):

    logger.debug(f"Event stream started | job_id={job_id} ")
    pubsub = redis_conn.redis_client.pubsub()
    pubsub.subscribe(job_id)
    try:
        pending_hitl = redis_conn.redis_client.get(f"hitl:{job_id}")
        if pending_hitl:
            review_data = json.loads(pending_hitl)
            if review_data.get('type') == 'bbox_review':
                update_job_progress(job_id, "processing", "process_plans")
                yield f"data: {json.dumps({'step': 'process_plans', 'status': 'processing'})}\n\n"
            elif review_data.get('type') == 'section_review':
                update_job_progress(job_id, "processing", "process_details")
                yield f"data: {json.dumps({'step': 'process_details', 'status': 'processing'})}\n\n"
            yield f"data: {json.dumps({'step': 'hitl_review', 'status': 'waiting_for_user', 'data': review_data})}\n\n"
        progress = get_job_progress(job_id)
        if progress:
            current_step = progress[2]
            status = progress[1]
            
            # Step order — send all previous steps as completed first
            step_order = ["classify", "process_plans", "process_details", "agent_4_merger"]
            
            if current_step in step_order and status == "processing":
                idx = step_order.index(current_step)
                # Mark all steps before current as completed
                for prev_step in step_order[:idx]:
                    yield f"data: {json.dumps({'step': prev_step, 'status': 'completed'})}\n\n"
            
            yield f"data: {json.dumps({'step': current_step, 'status': status})}\n\n"
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
            sheet_prefix = redis_conn.redis_client.get(f"sheet_prefix:{job_id}") or ""
            for thread_id, event in stream_estimation(job_id, file_path, output_dir,sheet_prefix=sheet_prefix):
    
                if job_id in cancelled_jobs:
                    logger.info(f"Job cancelled, stopping | job_id={job_id}")
                    cancelled_jobs.discard(job_id)
                    return
                logger.info(f"STREAM EVENT 👉 {event}")

                if not isinstance(event, dict):
                    continue

                if "__interrupt__" in event:
                    logger.info("INTERRUPT DETECTED")
                    logger.info(f"RAW INTERRUPT  {event['__interrupt__']}")
                    interrupt_obj = event["__interrupt__"]

                    if isinstance(interrupt_obj, tuple):
                        interrupt_obj = interrupt_obj[0]

                    review_data = interrupt_obj
                    while hasattr(review_data, "value"):
                        review_data = review_data.value
                    logger.info(f" CLEAN REVIEW DATA -> {review_data}")
                    logger.info(" SENDING HITL TO FRONTEND")

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
                    update_job_progress(job_id, "processing", node_name) 
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
            if job_id in cancelled_jobs:
                logger.info(f"Job cancelled, stopping | job_id={job_id}")
                cancelled_jobs.discard(job_id)
                return
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

    resume_payload=payload
    logger.info(f" RESUME STARTED | job_id={job_id}")
    logger.info(f" RESUME PAYLOAD -> {resume_payload}")
    def resume():
        pending = redis_conn.redis_client.get(f"hitl:{job_id}")
        hitl_type = None
        if pending:
            try:
                pending_data = json.loads(pending)
                hitl_type = pending_data.get("type")
                remaining_after_this = pending_data.get("remaining_after_this", 0)
            except:
                pass
        redis_conn.redis_client.delete(f"hitl:{job_id}")

        # Decide UI step based on which HITL we just left
        if hitl_type == "classify_review":
            next_ui_step = "process_plans"
        elif hitl_type == "bbox_review":
            next_ui_step = "process_plans" if remaining_after_this > 0 else "process_details"
        elif hitl_type == "section_review":
            next_ui_step = "process_details" if remaining_after_this > 0 else "agent_4_merger"
        else:
            next_ui_step = None

        # Mark previous step as done in DB and tell frontend
        if next_ui_step:
            step_order = ["classify", "process_plans", "process_details", "agent_4_merger"]
            idx = step_order.index(next_ui_step)
            # Mark all previous as completed
            for prev in step_order[:idx]:
                redis_conn.redis_client.publish(job_id, dumps({"step": prev, "status": "completed"}))
            update_job_progress(job_id, "processing", next_ui_step)
            redis_conn.redis_client.publish(job_id, dumps({"step": next_ui_step, "status": "processing"}))

        

  
        try:
            for thread_id, event in stream_estimation(
                job_id,
                None,
                None,
                command=Command(resume=resume_payload)
            ):
                
                if job_id in cancelled_jobs:
                        logger.info(f"Job cancelled during resume, stopping | job_id={job_id}")
                        cancelled_jobs.discard(job_id)
                        return
                
                logger.info(f" RESUME EVENT -> {event}")

                if not isinstance(event, dict):
                    continue

                if "__interrupt__" in event:
                    logger.info(" INTERRUPT AGAIN AFTER RESUME")
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

                for node_name, state_update in event.items():
                    logger.info(f" NEXT NODE AFTER RESUME -> {node_name}")
                    if node_name == "process_details" and isinstance(state_update, dict):
                        remaining = state_update.get("remaining_section_pages", [])
                        if len(remaining) == 0:
                            step_order = ["classify", "process_plans", "process_details", "agent_4_merger"]
                            for prev in step_order[:3]:
                                redis_conn.redis_client.publish(job_id, dumps({"step": prev, "status": "completed"}))
                            update_job_progress(job_id, "processing", "agent_4_merger")
                            redis_conn.redis_client.publish(job_id, dumps({"step": "agent_4_merger", "status": "processing"}))
                    
                    if node_name == "agent_4_merger":
                        step_order = ["classify", "process_plans", "process_details", "agent_4_merger"]
                        for prev in step_order[:3]:
                            redis_conn.redis_client.publish(job_id, dumps({"step": prev, "status": "completed"}))
                        update_job_progress(job_id, "processing", "agent_4_merger")
                        redis_conn.redis_client.publish(job_id, dumps({"step": "agent_4_merger", "status": "processing"}))
                   
            update_job_progress(job_id, "completed", "agent_4_merger")
            redis_conn.redis_client.publish(job_id, json.dumps({
                "step": "agent_4_merger",
                "status": "completed"
            }))
            logger.info(f"Job completed after resume | job_id={job_id}")
                   
        except Exception as e:
            if job_id in cancelled_jobs:
                logger.info(f"Job cancelled during resume, stopping | job_id={job_id}")
                cancelled_jobs.discard(job_id)
                return
            logger.error(f"Resume failed | job_id={job_id} | error={str(e)}")     

        
    threading.Thread(target=resume).start()

    return {"message": "resumed"}

@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str):
    return StreamingResponse(
        event_generator(job_id),
        media_type="text/event-stream"
    )