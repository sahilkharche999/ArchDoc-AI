import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
import json
# Import your graph builder (NOT the compiled app)
from src.workflow.workflows.estimation.graph import workflow

# 1. Setup Database
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# 2. Compile Graph
app = workflow.compile(checkpointer=memory)
# 3. Config (MUST match your previous run ID)
config = {"configurable": {"thread_id": "job_123"}}

# 4. "Time Travel" Logic
print("Rewinding state to start of Agent 2 (Plan Scanner)...")

# We update the state "as_node='process_details'".
# This tells LangGraph: "The 'process_details' node just finished."
# Since 'process_plans' comes after 'process_details', it will run next. 

app.update_state(
    config, 
    # You can optionally clear old plan data to force a fresh start
    # {"raw_plan_data": []}, 
    {},
    as_node="process_details"  # <--- This triggers next node ex. for 'process_plans' -> 'process_details' will be start
)
# process_text
# process_plans 
# process_details 

# 5. Run
print("Resuming execution...")
# Passing None tells it to continue from the state we just set 
print("Resuming execution...")

def pydantic_encoder(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)

for event in app.stream(None, config=config):
    for node_name, state_update in event.items():
        print(f"Running Node: {node_name}")
        
        # Optional: Print the result if it finishes 
        if node_name == "agent_4_merger":
            print("\n=== MERGER RESULT ===")
            print(json.dumps(state_update, indent=2, default=pydantic_encoder))
