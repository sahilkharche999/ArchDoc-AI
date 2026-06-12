from src.db.checkpoint import memory
from src.workflow.workflows.estimation.graph import workflow
from typing import Optional

# Compile once
app = workflow.compile(checkpointer=memory)


def stream_estimation(job_id: str, pdf_path: str, output_dir: str,command=None, sheet_prefix: str = "", gemini_api_key: Optional[str] = None):
    thread_id = job_id
    config = {"configurable": {"thread_id": thread_id}}
    if command:
        iterator = app.stream(command, config=config)
    else:
        initial_state = {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "gemini_api_key": gemini_api_key,
        "sheet_prefix": sheet_prefix,

        "page_map": {},
        "detail_library": {},
        "final_bill_of_materials": {},

        "floor_plan_images": [],
        "detected_details": [],

        # NOTE: remaining_pages / current_page / remaining_section_pages / current_section_page 
        # are intentionally NOT initialized here.
        # The nodes populate them lazily via an absence-based guard
        # (`if "remaining_pages" not in state:`). Initializing them would make
        # the key always-present, the guard would never fire, and NO pages
        # would process. Leave them absent.      

        # "remaining_pages": [],  # populated in node_process_plans
        # "current_page": None,   # set in node_process_plans

        # "remaining_section_pages": [], # populated in node_process_details
        # "current_section_page": None,  # set in node_process_details
        
        "temp_dependent_details": [],
        "temp_plan_like_details": [],
        }
        iterator = app.stream(initial_state, config=config)

    for event in iterator:
        yield thread_id, event
