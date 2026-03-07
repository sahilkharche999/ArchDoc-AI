from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
from src.service import stream_estimation
from src.workflow.common.logger import setup_logger
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.workflow.workflows.estimation.graph import workflow


logger = setup_logger(__name__)
router = APIRouter()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app_graph = workflow.compile(checkpointer=memory)

def event_generator(file_path):
    for thread_id, event in stream_estimation(file_path, "output_temp"):
        for node_name, state_update in event.items():
            payload = {
                "node": node_name
            }
            yield f"data: {json.dumps(payload)}\n\n"


@router.get("/jobs/stream")
def stream_job(file_path: str):
    logger.info("calling the estimation workflow ")
    return StreamingResponse(
        event_generator(file_path),
        media_type="text/event-stream"
    )

@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    config = {"configurable": {"thread_id": job_id}}

    snapshot = app_graph.get_state(config)

    if not snapshot.values:
        return {"error": "Job not found"}

    bom = snapshot.values.get("final_bill_of_materials", {})

    logger.info(f"Here is how the bom looks like : { bom.get("final_bill_of_materials", [])}")

    return {
        "job_id": job_id,
        "bom": bom.get("final_bill_of_materials", [])
    }
