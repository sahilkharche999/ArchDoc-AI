import uuid
from src.workflow.workflows.estimation.graph import workflow
from src.checkpoint import memory

# Compile once
app = workflow.compile(checkpointer=memory)

def stream_estimation(pdf_path: str, output_dir: str):

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

    for event in app.stream(initial_state, config=config):
        yield thread_id, event
        

def start_job(pdf_path: str, output_dir: str):

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

    for _ in app.stream(initial_state, config=config):
        pass

    return thread_id


def get_job_state(thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return app.get_state(config)