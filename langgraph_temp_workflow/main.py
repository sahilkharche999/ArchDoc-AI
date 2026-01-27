import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph_temp_workflow.workflows.estimation.graph import workflow  

# 1. Setup DB
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)

# 2. Compile the Main Graph WITH Memory
# Note: You usually compile in main.py if you want to inject the checkpointer dynamically
app = workflow.compile(checkpointer=memory)

# 3. Run with Thread ID
config = {"configurable": {"thread_id": "job_123"}}

print("--- STARTING PARENT WORKFLOW ---")
initial_state = {
        "pdf_path": "langgraph_temp_workflow/input.pdf",
        "output_dir": "output_temp",
        "page_map": {}, "detail_library": {}, "general_rules": "", "raw_plan_data": [], "final_bill_of_materials": {}
    }
# Use stream to see outputs step-by-step
for event in app.stream(initial_state, config=config):
    for node_name, state_update in event.items():
        print(f"Finished Node: {node_name}")
        # You can print specific state keys here to debug langgraph_temp_workflow/main.py