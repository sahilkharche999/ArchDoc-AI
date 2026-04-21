from src.db.checkpoint import memory
from src.workflow.workflows.estimation.graph import workflow

# Compile once
app = workflow.compile(checkpointer=memory)


def stream_estimation(job_id: str, pdf_path: str, output_dir: str,command=None):
    thread_id = job_id
    config = {"configurable": {"thread_id": thread_id}}
    if command:
        iterator = app.stream(command, config=config)
    else:
        initial_state = {
        "pdf_path": pdf_path,
        "output_dir": output_dir,
        "page_map": {},
        "detail_library": {},
        "general_rules": "",
        "raw_plan_data": [],
        "final_bill_of_materials": {}
        }
        iterator = app.stream(initial_state, config=config)

    for event in iterator:
        yield thread_id, event
