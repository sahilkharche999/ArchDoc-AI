import uuid
from src.workflow.workflows.estimation.graph import workflow
from src.checkpoint import memory

# Compile once
app = workflow.compile(checkpointer=memory)


def start_estimation(pdf_path: str, output_dir: str):

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "page_map": {},
        "detail_library": {},
        "general_rules": "",
        "raw_plan_data": [],
        "final_bill_of_materials": {}
    }

    final_result = None

    for event in app.stream(initial_state, config=config):
        for node_name, state_update in event.items():
            if "final_bill_of_materials" in state_update:
                final_result = state_update

    return thread_id, final_result