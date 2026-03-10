from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
from src.service import stream_estimation
from src.workflow.common.logger import setup_logger
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from src.workflow.workflows.estimation.graph import workflow
from src.workflow.common.utils import load_material_weights
from src.workflow.common.utils import normalize_material

logger = setup_logger(__name__)
router = APIRouter()
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
app_graph = workflow.compile(checkpointer=memory)

def event_generator(job_id,file_path):
    for thread_id, event in stream_estimation(job_id,file_path, "output_temp"):
        for node_name, state_update in event.items():
            payload = {
                "node": node_name
            }
            yield f"data: {json.dumps(payload)}\n\n"

@router.get("/jobs/stream")
def stream_job(job_id: str,file_path: str):
    logger.info("calling the estimation workflow ")
    return StreamingResponse(
        event_generator(file_path),
        media_type="text/event-stream"
    )

def enrich_bom_with_pricing(bom_items, material_lookup):

    for item in bom_items:

        material = normalize_material(item["material_size"])

        material_data = material_lookup.get(material, {})

        lb_per_ft = material_data.get("lb_per_ft", item.get("lb_per_ft", 0))
        price = material_data.get("price_per_lb", 0)

        item["lb_per_ft"] = lb_per_ft

        item["total_weight_lbs"] = item["total_linear_feet"] * lb_per_ft * item.get("quantity", 1)

        item["charge_per_lb"] = price

        item["total_cost"] = item["total_weight_lbs"] * price

    return bom_items


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    config = {"configurable": {"thread_id": job_id}}

    snapshot = app_graph.get_state(config)
    
    if not snapshot.values:
        return {"error": "Job not found"}

    bom_wrapper = snapshot.values.get("final_bill_of_materials", {})
    bom = bom_wrapper.get("final_bill_of_materials", [])

    EXCEL_PATH = "#1A Steel Estimator (2023).xlsx"
    MATERIAL_LOOKUP = load_material_weights(EXCEL_PATH)
    bom = enrich_bom_with_pricing(bom, MATERIAL_LOOKUP)
    return {
        "job_id": job_id,
        "bom": bom
    }
