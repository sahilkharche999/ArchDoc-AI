import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import json
# Import your graph builder (NOT the compiled app)
from src.workflow.workflows.estimation.graph import workflow

# 1. Setup Database
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)


app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "job_123"}}


app.update_state(
    config, 
    {},
    as_node="process_details"  # <--- This triggers next node ex. for 'process_plans' -> 'process_details' will be start
)
# process_text
# process_plans 
# process_details 
# agent 4

# 5. Run
print("Resuming execution...")

def pydantic_encoder(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


for event in app.stream(None, config=config):
    for node_name, state_update in event.items():
        print(f"Running Node: {node_name}")

        if node_name == "agent_4_merger":
            print("\n=== MERGER RESULT ===")
            print(json.dumps(state_update, indent=2, default=pydantic_encoder))

    